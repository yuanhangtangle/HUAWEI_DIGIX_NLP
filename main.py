from torch.optim import AdamW, Adam
from utils.dataset import DocDataset, ModelDataset, ModelDataLoader
from utils.utils import get_conifgs
import torch.nn as nn
from model.model import Model
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
    configPath = './config/debug_configs.json'
    BERT_VERSION, csv_file_path, \
    LABELED_BATCH_SIZE, UNLABELED_BATCH_SIZE, EPOCHS \
        = global_configs(config_path=configPath)

    doc_dataset = DocDataset(labeled=True, config_path=configPath)
    model_dataset = ModelDataset(doc_dataset)
    model_dataloader = ModelDataLoader(model_dataset, batch_size=LABELED_BATCH_SIZE, shuffle=True)
    net = Model()
    validator = Validator(model=net, dataloader=model_dataloader)
    loss_fn = nn.CrossEntropyLoss()

    ops = [
        Adam(net.none_bert_parameters(), lr=1e-3),
        AdamW(net.bert_parameters(), lr=2e-5)
        ]
    lr_schs = [torch.optim.lr_scheduler.ExponentialLR(ops[0], 0.99, verbose=True)]
    joint_op = JointOptimizers(optimizers=ops, lr_schedulers=lr_schs)

    trainer = Trainer(model=net, dataloader=model_dataloader, loss=loss_fn, joint_optimizers=joint_op, epochs=EPOCHS,
                      validator=validator, verbose=True)
    trainer.train()
    # training loop
