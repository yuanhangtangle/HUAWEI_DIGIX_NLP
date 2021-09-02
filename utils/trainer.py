from typing import Optional
import torch
from tqdm import tqdm
from utils.joint_optimizer import JointOptimizers
from utils.validator import Validator


class Trainer:
    def __init__(
            self,
            model,
            dataloader,
            loss,
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
        self.loss = loss
        self.validator = validator
        self.save_path = save_path
        self.ep = 0
        self.verbose = verbose

        if self.save_path is not None:
            assert validator is not None, "A `Validator` MUST be given if `save_path` is given"

    def _train_single_epoch(self, epoch):
        self.model.train()
        for batch, (xs, ys) in tqdm(enumerate(self.dataloader)):
            loss = self._train_single_batch(xs, ys)
            self.joint_optims.zero_grad()
            loss.backward()
            self.joint_optims.step()
            if self.verbose:
                print(f"epoch {epoch} batch {batch}: loss = {loss.item():.4f}")

        print(f"epoch {epoch}: loss = {loss.item():.4f}")
        self.ep += 1
        # if the given optimizers have lr_schedulers
        if hasattr(self.joint_optims, 'adjust_lr'):
            self.joint_optims.adjust_lr()
        # the given loss may adjust parameters
        if hasattr(self.loss, 'step'):
            self.loss.step()

    def _train_single_batch(self, xs, ys):
        pred = self.model(xs)
        loss = self.loss(pred, ys)
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
            'loss': self.loss
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
        super(ModelTrainer, self).__init__(model, dataloader, loss_fn, joint_optimizers, epochs, validator, save_path,
                                           verbose)

    def _train_single_epoch(self, epoch):
        self.model.train()
        for batch, (inp, att, l, wc, ys) in enumerate(self.dataloader):
            xs = ((inp, att, l), wc)
            loss = self._train_single_batch(xs)
            self.joint_optims.zero_grad()
            loss.backward()
            self.joint_optims.step()
            if self.verbose:
                print(f"epoch {epoch} batch {batch}: loss = {loss.item():.4f}")

        print(f"epoch {epoch}: loss = {loss.item():.4f}")
        self.joint_optims.adjust_lr()
