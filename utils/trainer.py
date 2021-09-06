from typing import Optional
import logging
from utils.logger import init_event_logger, DataLogger
from tqdm import tqdm
from collections import defaultdict
from utils.joint_optimizer import JointOptimizers
from utils.validator import Validator

# log


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
            log:bool = True,
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
        self.log = log
        self.verbose = verbose
        self.history = defaultdict(list)
        self.log_header = ['epoch', 'loss']
        if self.save_path is not None:
            assert validator is not None, "A `Validator` MUST be given if `save_path` is given"

        if self.log:
            init_event_logger()
            l = ['epoch', 'loss']

    def _train_epoch(self, epoch):
        self.model.train()
        for idx, batch in tqdm(enumerate(self.dataloader)):
            loss = self._train_batch(batch)
            self.joint_optims.zero_grad()
            loss.backward()
            self.joint_optims.step()
            if self.verbose:
                print(f"epoch {epoch} batch {batch}: loss = {loss.item():.4f}")
        return loss

    def _train_batch(self, batch):
        xs, ys = batch
        if hasattr(self.model, 'train_batch'):
            pred = self.model.train_batch(xs)
        else:
            pred = self.model(xs)
        loss = self.loss(pred, ys)
        return loss

    def train(self):
        for ep in range(self.ep, self.epochs):
            self.ep = ep
            loss = self._train_epoch(ep)
            print(f"epoch {ep}: loss = {loss.item():.4f}")
            self.ep += 1

            # if the given optimizers have lr_schedulers
            if hasattr(self.joint_optims, 'adjust_lr'):
                self.joint_optims.adjust_lr()

            # the given loss may adjust parameters
            if hasattr(self.loss, 'step'):
                self.loss.step()

            # print information, write history and save models
            if self.validator is not None:
                s, is_best = self.validator.score()
                self.history['val_score'].append(s)
                self.history['is_best'].append(is_best)
                print(f"epoch {ep} : validation score = {s}")
                if self.save_path is not None and is_best:
                    pass  # remain to save models

            # write history
            self.history['epoch'].append(self.ep)
            self.history['loss'].append(loss.item())

            for obj in [self.loss, self.joint_optims, self.validator]:
                if hasattr(obj, 'write_history'):
                    obj.write_history(self.history)

    '''
    def save_states(self):
        torch.save({
            'ep': self.ep,
            'epochs': self.epochs,
            'model_state_dict': self.model.state_dict(),
            'joint_optimizers_state_dict': self.joint_optims.state_dict(),
            'loss': self.loss
        }, self.save_path)
    '''


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

    def _train_epoch(self, epoch):
        self.model.train()
        for batch, (inp, att, l, wc, ys) in enumerate(self.dataloader):
            xs = ((inp, att, l), wc)
            loss = self._train_batch(xs)
            self.joint_optims.zero_grad()
            loss.backward()
            self.joint_optims.step()
            if self.verbose:
                print(f"epoch {epoch} batch {batch}: loss = {loss.item():.4f}")

        print(f"epoch {epoch}: loss = {loss.item():.4f}")
        self.joint_optims.adjust_lr()
