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
    corpus.drop(columns=['title'], inplace=True)
    return pd.merge(corpus, wc, on='id', how='inner')


class DocDataset:
    def __init__(
            self,
            config_path: str
    ):
        configs = get_configs(config_path)
        label_name = configs.label_name
        unlabeled_mark = configs.unlabeled_mark
        bert_version = configs.bert_version
        self.labeled = configs.labeled

        if configs.debug:
            csv_file_path = configs.csv_file_path
            ds = pd.read_csv(csv_file_path)
        else:
            corpus_json = configs.corpus_json
            wc_csv = configs.wc_csv
            corpus_cols = configs.corpus_cols
            ds = read_corpus_and_wc(corpus_json, wc_csv, corpus_cols)
        ds[label_name].fillna(unlabeled_mark, inplace=True)

        self.bert_version = bert_version
        self.preprocessor = Preprocessor(bert_version)
        self.wc_mms = MinMaxScaler()

        # map labels
        self.labels, self.label_to_index, self.index_to_label = \
            self.preprocessor.map_label_to_index(ds[label_name], unlabeled_mark)
        self.labels = torch.tensor(self.labels.to_numpy(), dtype=torch.long)  # .reshape(-1,1)

        # cut and pad docs
        self.bert_input_ids, self.bert_attention_mask, self.doc_length\
            = self.preprocessor.cut_and_pad_docs(ds.body)

        # transform wc
        ds[configs.wc_features_names] = \
            self.wc_mms.fit_transform(ds[configs.wc_features_names])
        self.writing_characteristic = torch.tensor(
            ds[[configs.category] + configs.wc_features_names].to_numpy(),
            dtype=torch.float
        )

        # deal with back-translation


class ModelDataset(Dataset):

    def __init__(self, doc_dataset: DocDataset):
        self.doc_dataset = doc_dataset
        self.length = len(self.doc_dataset)
        assert self.length > 0, "The given dataset can't be empty"

    def __len__(self):
        return self.doc_dataset.length

    def __getitem__(self, item):
        return (
            self.doc_dataset.bert_input_ids[item],
            self.doc_dataset.bert_attention_mask[item],
            self.doc_dataset.doc_length[item],
            self.doc_dataset.writing_characteristic[item],
            self.doc_dataset.labels[item]
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
