from model.subnets import SemanticSubnet, WCSubnet
import torch
import torch.nn as nn
from itertools import chain


class Model(nn.Module):

    def __init__(
            self,
            semantic_dim: int = 64,
            wc_dim: int = 64,
            num_classes: int = 10,
            mlp_dropout: float = 0.4
    ):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.semantic_subnet = SemanticSubnet()
        self.wc_subnet = WCSubnet()
        self.mlp = nn.Linear(
            in_features=semantic_dim + wc_dim,
            out_features=num_classes
        )
        self.mlp_dropout = nn.Dropout(p=mlp_dropout)

    def forward(self, xs):
        (se, wc) = xs
        bert_output = self.semantic_subnet(se)
        wc_output = self.wc_subnet(wc)
        _o = torch.cat((bert_output, wc_output), dim=1).view(bert_output.shape[0], -1)
        _o = self.mlp_dropout(_o)
        _o = self.mlp(_o)
        return _o

    def bert_parameters(self):
        return self.semantic_subnet.bert_parameters()

    def none_bert_parameters(self):
        return chain(
            self.semantic_subnet.none_bert_parameters(),
            self.wc_subnet.parameters(),
            self.mlp.parameters()
        )