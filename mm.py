import math

import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.query_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.key_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.value_proj = nn.Linear(config.hidden_size, config.hidden_size)

        assert config.hidden_size % config.num_heads == 0, "不能整除"
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.dropout = nn.Dropout(config.dropout_rate)
        self.all_head_size = self.num_heads * self.head_dim

    def transpose_dim(self, hidden_dim):
        bsz, seq, hidden = hidden_dim.size()
        hidden_dim = hidden_dim.resize(bsz, seq, self.num_heads, self.head_dim)
        return hidden_dim.permute(0, 2, 1, 3)

    def forward(self, hidden):
        query_layer = self.transpose_dim(self.query_proj(hidden))
        key_layer = self.transpose_dim(self.key_proj(hidden))
        value_layer = self.transpose_dim(self.value_proj(hidden))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.head_dim)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        return context_layer

