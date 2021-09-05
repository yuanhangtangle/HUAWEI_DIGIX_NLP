from transformers import BertModel
from torch import nn
import torch
import torch.nn.functional as F
from itertools import chain
from model.layers import Attention


class SemanticSubnet(nn.Module):
    def __init__(
            self,
            bert_version: str = 'bert-base-chinese',
            bert_embed_size: int = 768,
            bert_dropout: float = 0.1,
            lstm_hidden_size: int = 16,
            lstm_num_layers: int = 1,
            bidirectional: bool = True,
            out_dim: int = 16,
            lstm_dropout: float = 0.
            # debug:bool = False
    ):
        self.out_dim = out_dim
        super(SemanticSubnet, self).__init__()
        self.bert = BertModel.from_pretrained(bert_version)
        self.bert_dropout = nn.Dropout(p=bert_dropout)
        self.lstm = nn.LSTM(
            input_size=bert_embed_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bidirectional=bidirectional,
            dropout=lstm_dropout
        )
        self.mlp = nn.Linear(
            in_features=lstm_hidden_size * 2 if bidirectional else lstm_hidden_size,
            out_features=out_dim
        )

    def forward(self, xs):
        input_ids, attention_mask, doc_length = xs
        bs = len(doc_length)
        _sm = []
        for idx in range(bs):
            l = doc_length[idx]
            inp, att = input_ids[idx][0:l], attention_mask[idx][0:l]
            _o = self.bert(
                input_ids=inp,
                attention_mask=att,
            ).pooler_output
            # transform 2d output to 3d
            _o = _o.unsqueeze(0)
            _o = self.bert_dropout(_o)
            _o, (_, _) = self.lstm(_o)
            _o = _o[:, -1, :]  # the last output
            _sm.append(self.mlp(_o))
        _sm = torch.cat(_sm, dim=0)
        return _sm

    def bert_parameters(self):
        return self.bert.parameters()

    def none_bert_parameters(self):
        return chain(
            self.lstm.parameters(),
            self.mlp.parameters()
        )


class WCSubnet(nn.Module):

    def __init__(
            self,
            num_category: int = 31,
            num_numeral: int = 28,
            embed_dim: int = 128,
            num_layers: int = 3,
            hidden_size: int = 256,
            num_head: int = 4,
            out_dim: int = 64,
            drop_attn: float = 0.1,
            drop_res: float = 0.1
    ):
        super(WCSubnet, self).__init__()
        self.out_dim = out_dim
        # embed category
        self.embed = nn.Embedding(num_category, embed_dim)
        # embed numerical features
        self.nume_W = nn.Parameter(torch.randn(embed_dim, num_numeral))
        # attention layers
        self.attentions = nn.ModuleList(
            [Attention(embed_dim, hidden_size, num_head, drop_attn, drop_res)] +
            [Attention(hidden_size, hidden_size, num_head, drop_attn, drop_res) for i in range(num_layers - 1)]
        )
        # concat, mlp
        self.mlp = nn.Linear((num_numeral + 1) * hidden_size, out_dim)
        # relu

    def forward(self, x):
        # x = (batch, category + nume)
        bs = x.shape[0]
        cate, nume = x[:, 0].type(torch.long), x[:, 1:]  # cate.shape == (bs), nume.shape == (bs, nume)
        cate = self.embed(cate)  # cate.shape == (bs, em_d)
        nume = (nume.unsqueeze(1) * self.nume_W).transpose(-1, -2)  # nume.shape == (bs, nume, em_d)
        embed = torch.cat([cate.unsqueeze(-2), nume], dim=-2)  # embed.shape == (bs, nume + 1, em_d)
        attns = embed
        for attnLayers in self.attentions:
            attns, _ = attnLayers(attns)  # attns.shape == (bs, nume + 1, hidden_size)
        attns = attns.view(bs, -1)  # attns.shape == (bs, (nume + 1)*hidden_size)
        o = F.relu(self.mlp(attns))  # o.shape == (bs, out_dim)
        return o
