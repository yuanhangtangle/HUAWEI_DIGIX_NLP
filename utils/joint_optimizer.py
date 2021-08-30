from typing import Union

from utils.utils import load_optimizer, load_lr_scheduler


class JointOptimizers:
    def __init__(
            self,
            optimizers: Union[list, tuple, None] = None,
            lr_schedulers: Union[list, tuple, None] = None
    ):
        self.optims = optimizers if optimizers is not None else []
        self.scheds = lr_schedulers if lr_schedulers is not None else []

    def zero_grad(self):
        for op in self.optims:
            op.zero_grad()

    def step(self):
        for op in self.optims:
            op.step()

    def adjust_lr(self):
        for sch in self.scheds:
            sch.step()

    def state_dict(self):
        optimizers_state_dicts = [(optim.__class__, optim.state_dict()) for optim in self.optims]
        lr_schedulers_state_dicts = [(sch.__class__, sch.state_dict()) for sch in self.scheds]
        return optimizers_state_dicts, lr_schedulers_state_dicts

    def load_state_dict(self, joint_optimizers_state_dict, last_epoch):
        optimizers_state_dicts, lr_schedulers_state_dicts = joint_optimizers_state_dict
        self.optims = [], self.scheds = []

        # load optimizers
        for opt_class, state_dict in optimizers_state_dicts:
            self.optims.append(load_optimizer(opt_class, state_dict))

        # load lr_schedulers if needed
        for idx,  lr_sched_class, state_dict in enumerate(lr_schedulers_state_dicts):
            self.scheds.append(load_lr_scheduler(lr_sched_class, self.optims[idx], state_dict))