import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy

class TCCritic(nn.Module):
    def __init__(self, scheme, args):
        super(TCCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        # Set up network layers
        self.state_dim = int(np.prod(args.state_shape))

        self.input_dim = self.state_dim + self.n_agents * self.n_actions

        self.hiden_dim = args.hiden_dim_rate * self.input_dim
        
        self.embeding_layer = nn.Linear(self.input_dim, self.hiden_dim)

        self.pos_emb = PositionalEncoding(self.hiden_dim)
        self.layers = nn.ModuleList([EncoderLayer(args, self.hiden_dim) for _ in range(args.n_layers)])
        self.fc = nn.Linear(self.hiden_dim, 1)

    def forward(self, act, states, mask):
        '''
        act: [batch_size, src_len, n_agents, available_actions]
        states: [batch_size, src_len, states_num]
        mask: [batch_size, src_len, 1]
        '''

        mask = copy.deepcopy(mask)
        batch_size, src_len = act.size(0), act.size(1)
        act = act.reshape(batch_size, src_len, -1)
        enc_inputs = torch.cat([states, act], dim=2)

        enc_outputs = self.embeding_layer(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, hiden_dim]
        enc_self_attn_mask = mask.transpose(-1,-2).expand(batch_size, src_len, src_len) # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, hiden_dim], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        q = self.fc(enc_outputs)

        return q, enc_self_attns


class PositionalEncoding(nn.Module):
    def __init__(self, hiden_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, hiden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hiden_dim, 2).float() * (-math.log(10000.0) / hiden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, hiden_dim]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask==0, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, args, hiden_dim):
        super(MultiHeadAttention, self).__init__()
        self.hiden_dim = hiden_dim
        self.n_heads = args.n_heads
        self.d_k = args.d_k
        self.d_v = args.d_v
        self.W_Q = nn.Linear(self.hiden_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.hiden_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.hiden_dim, self.d_v * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.d_v, self.hiden_dim, bias=False)
    def forward(self, inputs, attn_mask):
        '''
        input_Q: [batch_size, len_q, hiden_dim]
        input_K: [batch_size, len_k, hiden_dim]
        input_V: [batch_size, len_v(=len_k), hiden_dim]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = inputs, inputs.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(inputs).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(inputs).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(inputs).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention(self.d_k)(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, hiden_dim]
        return nn.LayerNorm(self.hiden_dim).cuda()(output + residual), attn
        # return torch.relu(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, args, hiden_dim):
        super(PoswiseFeedForwardNet, self).__init__()
        self.hiden_dim = hiden_dim
        self.d_ff = args.d_ff
        self.fc = nn.Sequential(
            nn.Linear(hiden_dim, self.d_ff, bias=False),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Linear(self.d_ff, hiden_dim, bias=False)
        )
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, hiden_dim]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.hiden_dim).cuda()(output + residual) # [batch_size, seq_len, hiden_dim]
        # return torch.relu(output + residual) # [batch_size, seq_len, hiden_dim]

class EncoderLayer(nn.Module):
    def __init__(self, args, hiden_dim):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(args, hiden_dim)
        self.pos_ffn = PoswiseFeedForwardNet(args, hiden_dim)

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, hiden_dim]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, hiden_dim], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, hiden_dim]
        return enc_outputs, attn
