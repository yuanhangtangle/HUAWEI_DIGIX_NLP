from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
from utils.dataset import DocDataset, SemanticDataset, BatchGenerator
from utils.utils import get_conifgs
import torch.nn as nn
from model.subnets import SemanticSubnet
from utils.trainer import OptimizerYH, Trainer, SemanticTrainer
from utils.joint_optimizer import JointOptimizers
import torch


def global_configs(configPath):
    configs = get_conifgs(configPath)
    labeled_batch_size = configs['labeled_batch_size']
    unlabeled_batch_size = configs['unlabeled_batch_size']
    epochs = configs['epochs']

    return configs['bert_version'], configs['csv_file_path'], \
           labeled_batch_size, unlabeled_batch_size, \
           epochs


if __name__ == '__main__':
    configPath = './config/debug_se_configs.json'
    BERT_VERSION, csv_file_path, \
    LABELED_BATCH_SIZE, UNLABELED_BATCH_SIZE, EPOCHS\
        = global_configs(configPath = './config/debug_se_configs.json')

    doc_dataset = DocDataset(
        labeled=True,
        config_path=configPath
    )

    semantic_dataset = SemanticDataset(doc_dataset)
    labeled_dataloader = BatchGenerator(
        semantic_dataset,
        batch_size=LABELED_BATCH_SIZE, shuffle=True
    )

    net = SemanticSubnet()

    loss_fn = nn.CrossEntropyLoss()

    op = JointOptimizers([
        OptimizerYH(AdamW(net.bert_parameters(), lr=2e-5)),
        OptimizerYH(Adam(net.none_bert_parameters(), lr=1e-3))
    ])

    trainer = Trainer(model=net, dataloader=labeled_dataloader, loss=loss_fn, joint_optimizers=op, epochs=EPOCHS)
    trainer.train()
    # training loop
