import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional
from abc import ABC, abstractmethod


class TSALoss(nn.Module):

    def __init__(
            self,
            epochs: Optional[int] = 1,
            num_classes: Optional[int] = 1,
            last_epoch: int = -1,
            annealing: Optional[str] = 'linear',
            base_eta: Optional[float] = None
    ):
        super(TSALoss, self).__init__()
        self.epochs = epochs
        self.last_epoch = last_epoch
        self.annealing = annealing
        self.eta = 1
        self.num_classes = num_classes
        self._base_eta = 1 / num_classes if base_eta is None else base_eta
        self.alpha = 1

        self.adjust_params()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """

        :param x: of shape (batch_size, num_classes)
        :param y: of shape (batch_size)
        :return: TSA cross entropy loss
        """
        conf = F.softmax(x, dim=1).max(dim=1).values
        x = x[conf < self.eta]
        y = y[conf < self.eta]
        return F.cross_entropy(x, y)

    def adjust_params(self):
        self.last_epoch += 1

        if self.annealing is None:
            return
        elif self.annealing == 'linear':
            self.alpha = self.ep / self.epochs
        elif self.annealing == 'log':
            self.alpha = 1 - torch.exp(- self.ep / self.epochs * 5)
        elif self.annealing == 'exp':
            self.alpha = torch.exp((self.ep / self.epochs - 1) * 5)

        self.eta = self.alpha * (1 - self._base_eta) + self._base_eta

    def state_dict(self):
        return {
            'epochs': self.epochs,
            'last_epoch': self.last_epoch,
            'annealing': self.annealing,
            'eta': self.eta,
            'num_classes': self.num_classes,
            '_base_eta': self._base_eta,
            'alpha': self.alpha
        }

    def load_state_dict(self, state_dict: dict):
        for k, v in state_dict.items():
            self.__dict__[k] = v
