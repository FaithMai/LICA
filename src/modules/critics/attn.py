import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ATTNCritic(nn.Module):
    def __init__(self, scheme, args):
        super(ATTNCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.encoder_dim = args.encoder_dim
        
        self.output_type = "q"

        # Set up network layers
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim * (self.n_agents * self.n_actions+self.encoder_dim)
        self.hid_dim = args.mixing_embed_dim

        self.encoder = ActionEncoder(self.n_agents * self.n_actions, args)
        self.attention = Attention(args)
        

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                           nn.ReLU(),
                                           nn.Linear(self.embed_dim, self.embed_dim))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, self.hid_dim),
                                           nn.ReLU(),
                                           nn.Linear(self.hid_dim, self.hid_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.hid_dim)

        self.hyper_b_2 = nn.Sequential(nn.Linear(self.state_dim, self.hid_dim),
                               nn.ReLU(),
                               nn.Linear(self.hid_dim, 1))

    def forward(self, act, states):
        bs = states.size(0)
        states = states.reshape(-1, self.state_dim)
    
        act_his = act.clone()
        batch_size, src_len, n_agents, available_actions = act_his.size()
        his_init = act_his.new_zeros((batch_size, 1, n_agents, available_actions))
        act_his = torch.cat([his_init, act_his], dim=1)
        act_his = act_his.reshape(batch_size, src_len+1, -1)
        cs = []
        for idx in range(1, act_his.shape[1]):
            his = act_his[:, :idx, :]
            enc_output, s0 = self.encoder(his)
            c = self.attention(s0, enc_output)
            cs.append(c)
        
        cs = torch.cat(cs, dim=1)
        act = act.reshape(batch_size, -1, self.n_agents * self.n_actions)
        act = torch.cat([act, cs], dim=-1)
        
        action_probs = act.reshape(-1, 1, self.n_agents*self.n_actions+self.encoder_dim)
        w1 = self.hyper_w_1(states)
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents * self.n_actions+self.encoder_dim, self.hid_dim)
        b1 = b1.view(-1, 1, self.hid_dim)

        h = torch.relu(torch.bmm(action_probs, w1) + b1)

        w_final = self.hyper_w_final(states)
        w_final = w_final.view(-1, self.hid_dim, 1)

        h2 = torch.bmm(h, w_final)

        b2 = self.hyper_b_2(states).view(-1, 1, 1)

        q = h2 + b2

        q = q.view(bs, -1, 1)

        return q

class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.args = args

        self.attn = nn.Linear(args.encoder_dim+args.decoder_dim, args.decoder_dim, bias=False)
        self.v = nn.Linear(args.decoder_dim, 1, bias=False)

    def forward(self, s, enc_output):
        trasition_len = enc_output.shape[1]
        s = s.unsqueeze(1).repeat(1, trasition_len, 1)

        # energy = [batch_size, trasition_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim = 2)))
        
        # attention = [batch_size, trasition_len]
        attention = F.softmax(self.v(energy).squeeze(2), dim=1)
        attention = attention.unsqueeze(1)
        
        # c = [batch_size, 1, args.encoder_dim]
        c = torch.bmm(attention, enc_output)
        return c

class ActionEncoder(nn.Module):
    def __init__(self, input_shape, args):
        super(ActionEncoder, self).__init__()
        self.args = args

        self.rnn = nn.GRU(input_shape, args.encoder_dim, batch_first=True)
        self.fc = nn.Linear(args.encoder_dim, args.decoder_dim)

    def forward(self, inputs):
        enc_output, enc_hidden = self.rnn(inputs)
        # s = torch.tanh(self.fc(torch.cat((enc_hidden[:,-1,:], enc_hidden[:,-2,:]), dim = 1)))
        s = torch.tanh(self.fc(enc_hidden.transpose(0,1)[:,-1,:]))
        return enc_output, s







