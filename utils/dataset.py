from torch.utils.data import Dataset, DataLoader
from utils.utils import get_configs, read_json_to_dataframe
from utils.preprocessor import Preprocessor
from sklearn.preprocessing import MinMaxScaler
import torch
import pandas as pd
import numpy as np


def read_corpus_and_wc(corpus_json_path, wc_csv_path, columns=None) -> pd.DataFrame:
    wc = pd.read_csv(wc_csv_path)
    # cols = ['id', 'title', 'body', 'category', 'doctype']
    corpus = read_json_to_dataframe(corpus_json_path, columns=columns)
    corpus.body = corpus.title + '。' + corpus.body
    return pd.merge(corpus, wc, on='id', how='inner')


class ClassificationDataset:
    def __init__(
            self,
            config_path: str
    ):
        configs = get_configs(config_path)
        label_name = configs.label_name
        unlabeled_mark = configs.unlabeled_mark
        bert_version = configs.bert_version
        self.labeled = configs.labeled

        ds = read_corpus_and_wc(configs.corpus_json, configs.wc_csv)
        self.length = ds.shape[0]
        self.bert_version = bert_version
        self.prep = Preprocessor(bert_version)

        # map labels
        self.labels, self.label_to_index, self.index_to_label = \
            self.prep.map_label_to_index(ds[label_name], unlabeled_mark)
        self.labels = torch.tensor(self.labels.to_numpy(), dtype=torch.long)  # .reshape(-1,1)

        # cut and pad docs
        self.inp, self.att, self.n_doc = self.prep.cut_and_pad_docs(ds.body)

        # transform wc
        ds[configs.wc_names] = self.prep.fit_scale(ds[configs.wc_features_names])
        self.wc = torch.tensor(
            ds[[configs.category] + configs.wc_names].to_numpy(),
            dtype=torch.float
        )

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return (
            self.inp[item],
            self.att[item],
            self.n_doc[item],
            self.wc[item],
            self.labels[item]
        )


class backTransDataset(Dataset):
    def __init__(self):
class ModelDataset(Dataset):

    def __init__(self, doc_dataset: ClassificationDataset):
        self = doc_dataset
        self.length = len(self)
        assert self.length > 0, "The given dataset can't be empty"

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return (
            self.bert_input_ids[item],
            self.bert_attention_mask[item],
            self.doc_length[item],
            self.writing_characteristic[item],
            self.labels[item]
        )


class ModelDataLoader:
    def __init__(self, dataset: Dataset, batch_size, shuffle: bool = False):
        self.dataloader = DataLoader(dataset, batch_size, shuffle)

    def __iter__(self):
        self.iterator = iter(self.dataloader)
        return self

    def __next__(self):
        try:
            d = next(self.iterator)
            inp, att, l, wc, y = d
            return ((inp, att, l), wc), y
        except StopIteration:
            raise StopIteration
