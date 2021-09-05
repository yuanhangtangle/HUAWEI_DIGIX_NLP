import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, List, Tuple, Union
from collections import defaultdict


class ClassificationLoss(nn.Module):

    def __init__(
            self,
            epochs: Optional[int] = 1,
            num_classes: Optional[int] = 1,
            last_epoch: int = -1,
            annealing: Optional[str] = 'linear',
            base_eta: Optional[float] = None
    ):
        super(ClassificationLoss, self).__init__()
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
        ind = conf < self.eta
        x, y = x[ind], y[ind]
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

    def write_history(self, history: Union[dict, defaultdict]):
        history['annealing_eta'].append(self.eta)
        history['annealing_alpha'].append(self.alpha)


class ConsistencyLoss:

    def __init__(self, temp: float = 1, thresh: float = 0.5):
        super(ConsistencyLoss, self).__init__()
        self.thresh = thresh
        self.temp = temp

    def __call__(self, y_o, y_p):
        assert y_o.shape == y_p.shape
        conf = F.softmax(y_o, dim=1).max(dim=1).values
        ind = conf > self.thresh
        y_o, y_p = y_o[ind], y_p[ind]
        bs = y_o.shape[0]
        y_o = F.softmax(y_o / self.temp)
        y_p = F.log_softmax(y_p)
        return -torch.sum(y_o * y_p) / bs

    def state_dict(self):
        return {
            'thresh': self.thresh,
            'temp': self.temp
        }

    def load_state_dict(self, state_dict: dict):
        for k, v in state_dict.items():
            self.__dict__[k] = v


class JointLoss:

    def __init__(
            self,
            lamda: float = 1,
            epochs: Optional[int] = 1,
            num_classes: Optional[int] = 1,
            last_epoch: int = -1,
            annealing: Optional[str] = 'linear',
            base_eta: Optional[float] = None,
            temp: float = 1,
            thresh: float = 0.5,
    ):
        super(JointLoss, self).__init__()
        self.lamda = lamda
        self.class_loss = ClassificationLoss(
            epochs, num_classes, last_epoch, annealing, base_eta
        )
        self.cons_loss = ConsistencyLoss(temp, thresh)

    def __call__(self, preds: tuple, target):
        (y_pred, y_origin, y_pert) = preds
        return self.class_loss(y_pred, target) + self.lamda * self.cons_loss(y_origin, y_pert)

    def state_dict(self):
        return [
            self.class_loss.state_dict(),
            self.cons_loss.state_dict()
        ]

    def load_state_dict(self, state_dict: List[dict]):
        class_dict, cons_dict = state_dict
        self.class_loss.load_state_dict(class_dict)
        self.cons_loss.load_state_dict(cons_dict)

    def write_history(self, history):
        self.class_loss.write_history(history)
