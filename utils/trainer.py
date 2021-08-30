from typing import Optional
import torch

from utils.joint_optimizer import JointOptimizers
from utils.validator import Validator


class Trainer:
    def __init__(
            self,
            model,
            dataloader,
            loss_fn,
            joint_optimizers: JointOptimizers,
            epochs: int,
            validator: Optional[Validator] = None,
            save_path: Optional[str] = None,
            verbose: bool = False
    ):
        self.dataloader = dataloader
        self.model = model
        self.joint_optims = joint_optimizers
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.validator = validator
        self.save_path = save_path
        self.ep = 0
        self.verbose = verbose

        if self.save_path is not None:
            assert validator is not None, "A `Validator` MUST be given if `save_path` is given"

    def _train_single_epoch(self, epoch):
        self.model.train()
        for batch, (xs, ys) in enumerate(self.dataloader):
            loss = self._train_single_batch(xs, ys)
            self.joint_optims.zero_grad()
            loss.backward()
            self.joint_optims.step()
            if self.verbose:
                print(f"epoch {epoch} batch {batch}: loss = {loss.item():.4f}")

        print(f"epoch {epoch}: loss = {loss.item():.4f}")
        self.joint_optims.adjust_lr()

    def _train_single_batch(self, xs, ys):
        pred = self.model(xs)
        loss = self.loss_fn(pred, ys)
        return loss

    def train(self):
        for ep in range(self.ep, self.epochs):
            self._train_single_epoch(ep)
            self.ep = ep

            # print information and save models
            if self.validator is not None:
                s, is_best = self.validator.score()
                print(f"epoch {ep} : validation score = {s}")
                if self.save_path is not None and is_best:
                    pass  # remain to save models

    def save_states(self):
        torch.save({
            'ep': self.ep,
            'epochs': self.epochs,
            'model_state_dict': self.model.state_dict(),
            'joint_optimizers_state_dict': self.joint_optims.state_dict(),
            'loss': self.loss_fn
        }, self.save_path)


class ModelTrainer(Trainer):
    def __init__(
            self,
            model,
            dataloader,
            loss_fn,
            joint_optimizers: JointOptimizers,
            epochs: int,
            validator: Optional[Validator] = None,
            save_path: Optional[str] = None,
            verbose: bool = False
    ):
        super(ModelTrainer, self).__init__(
            model,
            dataloader,
            loss_fn,
            joint_optimizers,
            epochs,
            validator,
            save_path,
            verbose
        )

    def _train_single_epoch(self, epoch):
        self.model.train()
        for batch, (se, wc, ys) in enumerate(self.dataloader):
            loss = self._train_single_batch(se, wc, ys)
            self.joint_optims.zero_grad()
            loss.backward()
            self.joint_optims.step()
            if self.verbose:
                print(f"epoch {epoch} batch {batch}: loss = {loss.item():.4f}")

        print(f"epoch {epoch}: loss = {loss.item():.4f}")
        self.joint_optims.adjust_lr()

    def _train_single_batch(self, se, wc, ys):
        pred = self.model(se, wc)
        loss = self.loss_fn(pred, ys)
        return loss