import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW, Adam
from model.model import Model

from utils.dataset import ClassDataset, ClassDataLoader, BackTransDataset
from utils.dataset import BackTransDataLoader, JointLoader
from utils.utils import get_configs, read_json_to_dataframe
from utils.trainer import Trainer
from utils.joint_optimizer import JointOptimizers
from utils.validator import Validator
from utils.preprocessor import Preprocessor
from utils.loss import JointLoss


def read_corpus_and_wc(corpus_json_path, wc_csv_path, columns=None) -> pd.DataFrame:
    wc = pd.read_csv(wc_csv_path)
    # cols = ['id', 'title', 'body', 'category', 'doctype']
    corpus = read_json_to_dataframe(corpus_json_path, columns=columns)
    if 'title' in corpus.columns:
        corpus.body = corpus.title + '。' + corpus.body
    return pd.merge(corpus, wc, on='id', how='inner')


def load_data(configs):
    prep = Preprocessor(configs.bert_version)
    ds = read_corpus_and_wc(configs.corpus_json, configs.corpus_csv)
    prep.mms_fit(ds[configs.wc_names])

    class_dataset = ClassDataset(ds, prep, configs)
    class_loader = ClassDataLoader(class_dataset, configs.class_batch_size, True)
    if configs.verbose:
        print('load classification corpus and writing characteristic successfully')
    ds = read_corpus_and_wc(configs.aux_json, configs.aux_csv)
    bt_dataset = BackTransDataset(ds, prep, configs)
    bt_loader = BackTransDataLoader(bt_dataset, configs.aux_batch_size, True)
    joint_loader = JointLoader(class_loader, bt_loader)

    return joint_loader, class_loader, bt_loader


if __name__ == '__main__':
    config_path = './config/debug_configs.json'
    configs = get_configs(config_path)

    joint_loader, class_loader, bt_loader = load_data(configs)
    net = Model()
    validator = Validator(model=net, dataloader=class_loader)
    loss = JointLoss()

    ops = [Adam(net.none_bert_parameters(), lr=1e-3), AdamW(net.bert_parameters(), lr=2e-5)]
    lr_schs = [torch.optim.lr_scheduler.ExponentialLR(ops[0], 0.99, verbose=True)]
    joint_op = JointOptimizers(optimizers=ops, lr_schedulers=lr_schs)

    trainer = Trainer(
        model=net,
        dataloader=joint_loader,
        loss=loss,
        joint_optimizers=joint_op,
        epochs=configs.epochs,
        validator=validator,
        verbose=configs.verbose
    )
    trainer.train()
    # training loop


