from torch.utils.data import Dataset, DataLoader
from utils.preprocessor import Preprocessor
import torch


class ClassDataset(Dataset):
    def __init__(
            self,
            ds,
            prep,
            configs
    ):
        super(ClassDataset, self).__init__()
        self.length = ds.shape[0]

        self.labels, self.label_to_index, self.index_to_label = prep.map_label_to_index(ds.doctype, unlabeled_mark='')
        self.labels = torch.tensor(self.labels.to_numpy(), dtype=torch.long)  # .reshape(-1,1)
        self.inp, self.att, self.n_doc = prep.cut_and_pad_docs(ds.body)
        ds[configs.wc_names] = prep.mms_scale(ds[configs.wc_names])
        self.wc = torch.tensor(ds[['category'] + configs.wc_names].to_numpy(), dtype=torch.float)

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


class BackTransDataset(Dataset):
    def __init__(self, ds, prep: Preprocessor, configs):
        super(BackTransDataset, self).__init__()
        self.length = ds.shape[0]
        self.src_inp, self.src_att, self.src_n_doc = prep.cut_and_pad_docs(ds.src)
        self.dst_inp, self.dst_att, self.dst_n_doc = prep.cut_and_pad_docs(ds.dst)
        ds[configs.wc_names] = prep.mms_scale(ds[configs.wc_names])
        self.wc = torch.tensor(ds[[configs.category] + configs.wc_names].to_numpy(), dtype=torch.float)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return (
            self.src_inp[item], self.src_att[item], self.src_n_doc[item],
            self.dst_inp[item], self.dst_att[item], self.dst_n_doc[item],
            self.wc[item]
        )


class ClassDataLoader:
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


class BackTransDataLoader:
    def __init__(self, dataset: Dataset, batch_size, shuffle: bool = False):
        self.dataloader = DataLoader(dataset, batch_size, shuffle)

    def __iter__(self):
        self.iterator = iter(self.dataloader)
        return self

    def __next__(self):
        try:
            d = next(self.iterator)
            src_inp, src_att, src_l, dst_inp, dst_att, dst_l, wc = d
            return ((src_inp, src_att, src_l), wc), ((dst_inp, dst_att, dst_l), wc)
        except StopIteration:
            raise StopIteration


class JointLoader:
    def __init__(self, class_loader, back_trans_loader):
        self.class_loader = class_loader
        self.bt_loader = back_trans_loader

    def __iter__(self):
        self.class_iterator = iter(self.class_loader)
        self.bt_iterator = iter(self.bt_loader)
        return self

    def __next__(self):
        try:
            x, y = next(self.class_iterator)
            in1, in2 = next(self.bt_iterator)
            return (x, in1, in2), y
        except StopIteration:
            raise StopIteration
