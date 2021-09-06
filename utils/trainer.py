from typing import Optional
import logging
from utils.logger import get_event_logger, DataLogger
from tqdm import tqdm
from collections import defaultdict
from utils.joint_optimizer import JointOptimizers
from utils.validator import Validator


logger = get_event_logger()


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
            log: bool = True
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
        self.history = defaultdict(list)
        self.log_header = ['epoch', 'loss']
        self.event_logger = None
        self.data_logger = None

        if self.save_path is not None:
            assert validator is not None, "A `Validator` MUST be given if `save_path` is given"

        if self.log:
            header = self.log_header[:]
            for obj in [self.loss, self.joint_optims, self.validator]:
                if hasattr(obj, 'log_header'):
                    header.extend(obj.log_header)
            self.data_logger = DataLogger(header)

    def _train_epoch(self, epoch):
        self.model.train()
        for idx, batch in enumerate(self.dataloader):
            loss = self._train_batch(batch)

            if loss.item() == 0.:
                logger.debug('NO sample used to compute loss')
                continue

            self.joint_optims.zero_grad()

            logger.debug("back-propagating ...")
            loss.backward()

            logger.debug('back-propagation successful')
            logger.debug(f"epoch {epoch} batch {idx}: loss = {loss.item():.4f}")

            self.joint_optims.step()

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
        logger.info('start training ...')
        for ep in range(self.ep, self.epochs):
            self.ep = ep
            loss = self._train_epoch(ep)
            self.ep += 1

            # if the given optimizers have lr_schedulers
            if hasattr(self.joint_optims, 'adjust_lr'):
                self.joint_optims.adjust_lr()

            # the given loss may adjust parameters
            if hasattr(self.loss, 'adjust_params'):
                self.loss.adjust_params()

            # print information, and save models
            if self.validator is not None:
                d_val = self.validator.epoch_info()
                s, is_best = d_val['val_score'], d_val['is_best']
                if self.save_path is not None and is_best:
                    pass  # remain to save models

            # write history
            d_epoch = {'epoch': self.ep, 'loss': loss.item()}
            d_epoch.update(d_val)
            for obj in [self.loss, self.joint_optims]:
                if hasattr(obj, 'epoch_info'):
                    d_epoch.update(obj.epoch_info())
            self.write_history(d_epoch)

            # write log
            if self.log:
                self.data_logger.log_data(d_epoch)
                logger.info(f"epoch: {self.ep :3d} loss: {loss.item():.3f} val_score: {s:.3f}")
        logger.info('finish training ...')

    def write_history(self, hist_d: dict):
        for k, v in hist_d.items():
            self.history[k].append(v)

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
