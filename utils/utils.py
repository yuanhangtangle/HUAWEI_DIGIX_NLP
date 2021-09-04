from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import json
from typing import Optional

def get_configs(config_path):
    with open(config_path, 'r') as f:
        _config = json.load(f)
    assert isinstance(_config, dict), "configs must be stored in a `dict`"
    return Configs(_config)


def generate_samples(x, y, n):
    _, xx, _, _ = train_test_split(x, y, test_size=n)
    return xx


def read_json_to_dataframe(path: str, columns=None) -> pd.DataFrame:
    ds = []
    with open(path, 'r') as f:
        for idx, line in enumerate(f):
            l = []
            line = json.loads(line)
            for k in line.keys():
                l.append(line[k])
            ds.append(l)
    ds = pd.DataFrame(ds)
    if columns is not None:
        ds.columns = columns
    else:
        ds.columns = line.keys()
    return ds


def load_optimizer(opt_class, state_dict):
    opt = opt_class(1e-3)
    opt.load_state_dict(state_dict)
    return opt


def load_lr_scheduler(lr_sched_class, opt, state_dict):
    sched = lr_sched_class(opt, 0.9)
    sched.load_state_dict(state_dict)
    return sched


class Configs:
    def __init__(self, configs: dict):
        for k, v in configs:
            self.__dict__[k] = v

if __name__ == '__main__':
    cols = ['id', 'title', 'body', 'category', 'doctype']
    corpus_p = '/home/yuanhang/HUAWEI_DIGIX/data/doc_quality_data_train.json'
    wc_p = '/home/yuanhang/HUAWEI_DIGIX/data/train_wc.csv'
    d = read_corpus_and_wc(corpus_p, wc_p, columns=cols)
