from torchmetrics.functional import auroc, accuracy, f1
from torch.utils.data import Dataset
from torch import nn
import torch


def micro_accuracy(preds, ys, num_classes):
    return accuracy(preds, ys, average='micro', num_classes=num_classes)


def macro_accuracy(preds, ys, num_classes):
    return accuracy(preds, ys, average='macro', num_classes=num_classes)


def my_auroc(preds, ys, num_classes):
    return auroc(preds, ys, num_classes=num_classes)


def micro_f1(preds, ys, num_classes):
    return f1(preds, ys, num_classes=num_classes, average='micro')


def macro_f1(preds, ys, num_classes):
    return f1(preds, ys, num_classes=num_classes, average='macro')


class Validator:

    def __init__(self, model: nn.Module, dataloader, score='auroc'):
        self.model = model
        self.num_classes = self.model.num_classes
        self.dataloader = dataloader
        self._best_score = -1
        self.log_header = ['is_best', 'val_score']
        score_func_map = {
            'f1': micro_f1,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'acc': micro_accuracy,
            'micro_acc': micro_accuracy,
            'macro_acc': macro_accuracy,
            'auroc': my_auroc
        }
        self.score_func = score_func_map[score]

    def score(self):
        preds, ys = self.infer()
        s = self.score_func(preds, ys, self.num_classes)
        is_best = False
        if s >= self._best_score:
            self._best_score = s
            is_best = True
        return s, is_best

    def infer(self):
        self.model.eval()
        preds, y = [], []
        with torch.no_grad():
            for xs, ys in enumerate(self.dataloader):
                preds.append(self.model(xs))
                y.append(ys)
        return torch.cat(preds, dim=0), torch.cat(y, dim=-1)
