from torch.utils.data import Dataset, DataLoader
from utils.utils import get_conifgs
from utils.preprocessor import Preprocessor
from sklearn.preprocessing import MinMaxScaler
import torch
import pandas as pd
import numpy as np


# we can build a super class that loads configs
# we can restrict the length of docs and sentences to represent
# the docs as a tensor instead of list

class DocDataset:
    def __init__(
            self,
            labeled: bool,
            config_path: str
    ):
        configs = get_conifgs(config_path)
        label_name = configs['label_name']
        unlabeled_mark = configs['unlabeled_mark']
        csv_file_path = configs['csv_file_path']
        bert_version = configs['bert_version']

        self.bert_version = bert_version
        self.preprocessor = Preprocessor(bert_version)
        self.labeled = labeled
        self.wc_mms = MinMaxScaler()
        ds = pd.read_csv(csv_file_path)
        self.length = len(ds)

        # sep labeled and unlabeled part
        if labeled:
            ds = ds[ds[label_name] != unlabeled_mark]
        else:
            ds = ds[ds[label_name] == unlabeled_mark]

        ds['body'] = ds['title'] + '。' + ds['body']  # add title to body
        doc_seq = ds['body'].apply(self.preprocessor.cut_doc)
        ds[configs['wc_features_names']] = self.wc_mms.fit_transform(ds[configs['wc_features_names']])
        self.labels, self.label_to_index, self.index_to_label = \
            self.preprocessor.map_label_to_index(ds[label_name], '')

        bert_inputs = self.preprocessor.bert_tokenize(doc_seq)

        self.labels = torch.tensor(self.labels.to_numpy(), dtype=torch.long)  # .reshape(-1,1)
        input_ids = list(bert_inputs.input_ids)
        attention_mask = list(bert_inputs.attention_mask)
        self.bert_input_ids, self.bert_attention_mask, self.doc_length\
            = self.preprocessor.pad_docs(input_ids, attention_mask)
        self.writing_characteristic = torch.tensor(
            ds[['category'] + configs['wc_features_names']].to_numpy(),
            dtype=torch.float
        )

    def __len__(self):
        return self.length


class SemanticDataset(Dataset):
    def __init__(
            self,
            doc_dataset: DocDataset
    ):
        self.doc_dataset = doc_dataset

    def __len__(self):
        return self.doc_dataset.length

    def __getitem__(self, item):
        x = (
            self.doc_dataset.bert_input_ids[item],
            self.doc_dataset.bert_attention_mask[item],
        )
        y = self.doc_dataset.labels[item]
        return x, y


class WCDataset(Dataset):
    def __init__(
            self,
            doc_dataset: DocDataset
    ):
        self.doc_dataset = doc_dataset

    def __len__(self):
        return self.doc_dataset.length

    def __getitem__(self, item):
        x = self.doc_dataset.writing_characteristic[item]
        y = self.doc_dataset.labels[item]
        return x, y


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


class BatchGenerator:

    def __init__(self, dataset, batch_size: int = 1, shuffle: bool = False):
        self.dataset = dataset
        self.length = len(self.dataset)
        self.batch_size = batch_size
        assert self.length > 0, "The given dataset can't be empty"
        self.indices = list(range(self.length))
        self.idx = 0
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.length:
            raise StopIteration
        _indices = self.indices[self.idx: self.idx + self.batch_size]
        self.idx += self.batch_size
        xs = [self.dataset[i][0] for i in _indices]
        ys = [self.dataset[i][1] for i in _indices]
        return xs, ys


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
