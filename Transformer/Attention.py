import math
import torch
from torch import nn 


def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    # X shape (batch_size, max_len)
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d) # Q * K.T / sqrt(d), 使用bmm是因为有batch_size
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.num_heads = num_heads

    def forward(self, queries, keys, values, valid_lens):
        # queries shape (batch_size, num_steps, query_size)
        # self.W_q shape (batch_size, num_steps, hidden_size)
        # transpose shape (batch_size * num_heads, num_steps, hidden_size / num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # output shape (batch_size * num_heads, num_steps, hidden_size / num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # X shape (batch_size, num_steps, hidden_size)
    
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # X shape (batch_size, num_steps, num_heads, hidden_size / num_heads)

    X = X.permute(0, 2, 1, 3)
    # X shape (batch_size, num_heads, num_steps, hidden_size / num_heads)

    return X.reshape(-1, X.shape[2], X.shape[3])
    # X shape (batch_size * num_heads, num_steps, hidden_size / num_heads)


def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    # X shape (batch_size * num_heads, num_steps, hidden_size / num_heads)
    
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    # X shape (batch_size, num_heads, num_steps, hidden_size / num_heads)
    
    X = X.permute(0, 2, 1, 3)
    # X shape (batch_size, num_steps, num_heads, hidden_size / num_heads)
    
    return X.reshape(X.shape[0], X.shape[1], -1)
    # X shape (batch_size, num_steps, hidden_size)