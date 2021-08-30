from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import json


def get_conifgs(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


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
    return ds


def load_optimizer(opt_class, state_dict):
    opt = opt_class(1e-3)
    opt.load_state_dict(state_dict)
    return opt


def load_lr_scheduler(lr_sched_class, opt, state_dict):
    sched = lr_sched_class(opt, 0.9)
    sched.load_state_dict(state_dict)
    return sched
