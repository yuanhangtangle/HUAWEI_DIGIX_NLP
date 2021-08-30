from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
from utils.dataset import DocDataset, WCDataset
from utils.utils import get_conifgs
import torch.nn as nn
from model.subnets import WCSubnet
from utils.trainer import Trainer
from utils.joint_optimizer import JointOptimizers
from utils.validator import Validator
import torch


def global_configs(config_path):
    configs = get_conifgs(config_path)
    labeled_batch_size = configs['labeled_batch_size']
    unlabeled_batch_size = configs['unlabeled_batch_size']
    epochs = configs['epochs']

    return configs['bert_version'], configs['csv_file_path'], \
           labeled_batch_size, unlabeled_batch_size, \
           epochs


if __name__ == '__main__':
    configPath = './config/debug_wc_configs.json'
    BERT_VERSION, csv_file_path, \
    LABELED_BATCH_SIZE, UNLABELED_BATCH_SIZE, EPOCHS \
        = global_configs(config_path=configPath)

    doc_dataset = DocDataset(labeled=True, config_path=configPath)
    wc_dataset = WCDataset(doc_dataset)
    wc_dataloader = DataLoader(wc_dataset, batch_size=LABELED_BATCH_SIZE, shuffle=True)

    net = WCSubnet()
    validator = Validator(model=net, dataset=wc_dataset)
    loss_fn = nn.CrossEntropyLoss()

    op = Adam(net.parameters(), lr=1e-3)
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(op, 0.99, verbose=True)
    joint_op = JointOptimizers([op], [lr_sch])

    trainer = Trainer(
        model=net,
        dataloader=wc_dataloader,
        loss_fn=loss_fn,
        joint_optimizers=joint_op,
        epochs=EPOCHS,
        validator=validator
    )
    trainer.train()
    # training loop
