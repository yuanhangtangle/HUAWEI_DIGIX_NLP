import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProductAttention(nn.Module):

    def __init__(self, temperature: float = 1, dropout: float = 0.1):
        super(DotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        :param q: (***, len_q, d_q), query sequences, *** can be batch, heads,etc;
        :param k: (***, len_k, d_k), key sequences, *** can be batch, heads,etc; NOTE that d_k == d_q
        :param v: (***, len_v, d_v), value sequences, *** can be batch, heads,etc; NOTE that len_v == len_k
        :return: torch tensor of shape (***, len_v, d_v) computed by dot-product attention mechanism.
        """

        attn = torch.matmul(q, k.transpose(-1, -2))  # attn.shape == (***, len_q, len_k)
        attn = self.dropout(F.softmax(attn, dim=-1))  # attn.shape == (***, len_q, len_k)
        attned_v = torch.matmul(attn, v)  # attned_v.shape == (***, len_q, d_v)
        return attn, attned_v


class Attention(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            hidden_size: int = 256,
            num_head: int = 4,
            drop_attn: float = 0.1,
            drop_res: float = 0.1
    ):
        self.in_size = embed_dim
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.single_hidden_size = hidden_size // num_head
        super(Attention, self).__init__()
        assert hidden_size % num_head == 0, "`hidden_size` must be a multiple of `num_head`"

        self.W = nn.Linear(embed_dim, hidden_size)
        self.dpAttn = DotProductAttention(dropout=drop_attn)
        self.fc = nn.Linear(embed_dim, hidden_size)
        self.drop_res = nn.Dropout(p=drop_res)

    def forward(self, x):
        """
        :param x: (batch_size, in_length, embed_dim)
        :return: (batch_size, in_length, hidden_size)
        """
        bs, in_length, in_size = x.shape[0], x.shape[1], x.shape[2]
        res = x
        
        x = self.W(x)  # (batch_size, in_length, hidden_size)
        x = x.view(bs, in_length, self.num_head, self.single_hidden_size)
        x = x.transpose(1, 2)
        attn, attned_x = self.dpAttn(x, x, x) # (bs, num_head, in_len, shs)
        attned_x = attned_x.transpose(1, 2).reshape(bs, in_length, -1)  # (bs, in_len, hidden_size)
        
        # add residue
        res = self.drop_res(self.fc(res)) # (bs, in_len, hidden_size)
        o = F.relu(attned_x + res) # (bs, in_len, hidden_size)
        
        return o, attn
