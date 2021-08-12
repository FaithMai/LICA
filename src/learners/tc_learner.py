import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.tc import TCCritic
from components.action_selectors import multinomial_entropy
from utils.rl_utils import build_td_lambda_targets
import torch as th
import torch.nn as nn
from torch.optim import RMSprop, Adam


class TCLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_episode = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.log_stats_t_agent = -self.args.learner_log_interval - 1

        self.critic = TCCritic(scheme, args)
        self.critic.apply(self.weights_init)
        self.target_critic = copy.deepcopy(self.critic)

        self.agent_params = list(self.mac.parameters())
        self.critic_params = list(self.critic.parameters())
        self.params = self.agent_params + self.critic_params

        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.critic_lr)

        self.entropy_coef = args.entropy_coef
    
    def weights_init(self, m):
        if isinstance(m, nn.Linear):                                           
            # nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.xavier_normal_(m.weight)
            # nn.init.constant_(m.bias, 0)
    
    # def show_grad_info(self, m):
    #     f = open('./results/tc_grad.txt', 'a+')
    #     for name, weight in m.named_parameters():
	# 		# print("weight:", weight) # 打印权重，看是否在变化
    #         if weight.requires_grad:
	# 			# print("weight:", weight.grad) # 打印梯度，看是否丢失
	# 			# 直接打印梯度会出现太多输出，可以选择打印梯度的均值、极值，但如果梯度为None会报错
    #             f.write('name:'+name+'\n')
    #             f.write("weight.grad: mean:"+str(round(weight.grad.mean().cpu().item(), 4))+' min:'+str(round(weight.grad.min().cpu().item(), 4))+' max:'+str(round(weight.grad.max().cpu().item(), 4))+'\n')
    #     f.write('\n\n\n')
    #     f.flush()
    #     f.close()

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]

        # Calculate action policy distribution and entropy
        mac_out = []
        mac_out_entropy = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t, return_logits=True)
            agent_entropy = multinomial_entropy(agent_outs).mean(dim=-1, keepdim=True)
            agent_probs = th.nn.functional.softmax(agent_outs, dim=-1)
            mac_out.append(agent_probs)
            mac_out_entropy.append(agent_entropy)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        mac_out_entropy = th.stack(mac_out_entropy, dim=1)

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0

        # Mix action probability and state to estimate joint Q-value
        mix_loss, enc_self_attns = self.critic(mac_out, batch["state"][:, :-1], mask)

        mask = mask.expand_as(mix_loss)
        entropy_mask = copy.deepcopy(mask)

        mix_loss = (mix_loss * mask).sum() / mask.sum()

        # Adaptive Entropy Regularization
        entropy_loss = (mac_out_entropy * entropy_mask).sum() / entropy_mask.sum()
        entropy_ratio = self.entropy_coef / entropy_loss.item()

        mix_loss = - mix_loss - entropy_ratio * entropy_loss

        # Optimise agents
        self.agent_optimiser.zero_grad()
        mix_loss.backward()

        # if t_env - self.log_stats_t_agent >= self.args.learner_log_interval:
        #     self.show_grad_info(self.critic)
        #     self.show_grad_info(self.mac.agent)

        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        if t_env - self.log_stats_t_agent >= self.args.learner_log_interval:
            self.logger.log_stat("mix_loss", mix_loss.item(), t_env)
            self.logger.log_stat("entropy", entropy_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.log_stats_t_agent = t_env


    def train_critic_td(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"]
        actions = batch["actions_onehot"]
        terminated = batch["terminated"].float()
        mask = batch["filled"].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Optimise critic
        target_q_vals, target_enc_self_attns = self.target_critic(actions, batch["state"], mask)
        target_q_vals = target_q_vals[:, :]

        # Calculate td-lambda targets
        '''
        rewards: [batch_size, src_len, 1]
        terminated: [batch_size, src_len, 1]
        mask: [batch_size, src_len, 1]
        target_q_vals: [batch_size, src_len, 1]
        '''

        targets = build_td_lambda_targets(rewards, terminated, mask, target_q_vals, self.n_agents, self.args.gamma, self.args.td_lambda)

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_t_mean": [],
        }

        mask = mask[:, :-1]

        q_t, enc_self_attns = self.critic(actions[:, :-1], batch["state"][:, :-1], mask)

        td_error = (q_t - targets.detach())

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()
        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()
        self.critic_training_steps += 1

        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm)
        mask_elems = mask.sum().item()
        running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
        running_log["q_t_mean"].append((q_t * mask).sum().item() / mask_elems)
        running_log["target_mean"].append((targets * mask).sum().item() / mask_elems)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(running_log["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_t_mean", "target_mean"]:
                self.logger.log_stat(key, sum(running_log[key])/ts_logged, t_env)
            self.log_stats_t = t_env

        # Update target critic
        if (self.critic_training_steps - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = self.critic_training_steps

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
