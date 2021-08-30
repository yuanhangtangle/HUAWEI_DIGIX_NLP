from torchmetrics.functional import auroc, accuracy, f1
from torch.utils.data import Dataset
from torch import nn
import torch


class Validator:

    def __init__(self, model: nn.Module, dataset: Dataset, score='auroc'):
        self.model = model
        self.num_classes = self.model.out_dim
        self._x, self._y = dataset[:]
        self._best_score = -1
        score_func_map = {
            'f1': self.micro_f1,
            'micro_f1': self.micro_f1,
            'macro_f1': self.macro_f1,
            'acc': self.micro_accuracy,
            'micro_acc': self.micro_accuracy,
            'macro_acc': self.macro_accuracy,
            'auroc': self.auroc
        }
        self.score_func = score_func_map[score]

    def score(self):
        s = self.score_func()
        is_best = False
        if s >= self._best_score:
            self._best_score = s
            is_best = True
        return s, is_best

    def infer(self):
        self.model.eval()
        with torch.no_grad():
            preds = self.model.forward(self._x)
        return preds

    def micro_accuracy(self):
        preds = self.infer()
        return accuracy(preds, self._y, average='micro', num_classes=self.num_classes)

    def macro_accuracy(self):
        preds = self.infer()
        return accuracy(preds, self._y, average='macro', num_classes=self.num_classes)

    def auroc(self):
        preds = self.infer()
        return auroc(preds, self._y, num_classes=self.num_classes)

    def micro_f1(self):
        preds = self.infer()
        return f1(preds, self._y, num_classes=self.num_classes, average='micro')

    def macro_f1(self):
        preds = self.infer()
        return f1(preds, self._y, num_classes=self.num_classes, average='macro')
