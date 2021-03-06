import re
import torch

import pandas as pd
import numpy as np

from transformers import BertTokenizer
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Optional
from utils.logger import get_event_logger

SEP = 102
PAD = 0
CLS = 101
logger = get_event_logger()


class Preprocessor:
    def __init__(
            self,
            bert_version: Optional[str] = None,
            max_sent_length: int = 16,
            max_doc_length: int = 16,
            truncation: bool = True,
    ):
        self.mms = MinMaxScaler()
        self.mms_fitted_ = False
        self.max_sent_length = max_sent_length
        self.max_doc_length = max_doc_length
        self.truncation = truncation
        self.bert_version = bert_version
        self.tokenizer = None
        if bert_version is not None:
            self.load_pretrained_tokenizer(self.bert_version)

    def load_pretrained_tokenizer(self, bert_tokenizer_version):
        logger.debug("loading bert tokenizer ...")
        self.tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_version)
        logger.debug("bert tokenizer loaded successfully")

    def sep_label_unlabel(self, labels, unlabeled_mark, *dfs):
        labeled = [], unlabeled = []
        for df in dfs:
            labeled.append(df[labels != unlabeled_mark])
            unlabeled.append(df[labels == unlabeled_mark])
        labeled_labels = labels[labels != unlabeled_mark]
        return labeled_labels, tuple(labeled), tuple(unlabeled)

    def get_label_dict(self, labels, unlabeled_mark=None) -> Tuple[dict, dict]:
        l = labels.unique()
        if unlabeled_mark is not None:
            l = l[l != unlabeled_mark]
            label_to_index = {b: a for a, b in enumerate(l)}
            label_to_index[unlabeled_mark] = len(label_to_index)
        else:
            label_to_index = {b: a for a, b in enumerate(l)}
        index_to_label = {b: a for (a, b) in label_to_index.items()}
        return label_to_index, index_to_label

    def map_label_to_index(self, labels, unlabeled_mark=None):
        '''
        map labels to indices to facilitate model training
        :param labels: pd.Series, list of labels
        :return:
        '''
        label_to_index, index_to_label = self.get_label_dict(labels, unlabeled_mark)
        labels = labels.apply(lambda x: label_to_index[x])
        return labels, label_to_index, index_to_label

    def cut_doc(self, para):
        '''
        remain to modify:
        1. maybe we can delete all the punctuations
        '''
        para = re.sub('\s*([?????????;\?])([^??????])\s*', r"\1\n\2", para)  # ??????????????????
        para = re.sub('\s*(\.{6})([^??????])\s*', r"\1\n\2", para)  # ???????????????
        para = re.sub('\s*(\???{2})([^??????])\s*', r"\1\n\2", para)  # ???????????????
        para = re.sub('\s*([?????????\?][??????])([^????????????\?])\s*', r'\1\n\2', para)
        para = re.sub(r'\s+', r'\n', para)
        para = para.rstrip()  # ????????????????????????\n????????????
        return para.split("\n")

    def bert_tokenize(self, doc_seq) -> pd.DataFrame:
        input_ids = []
        attention_mask = []
        token_type_ids = []
        for d in doc_seq:
            tk = self.tokenizer(
                d, return_tensors='pt', padding=True, max_length=self.max_sent_length,
                truncation=self.truncation
            )
            input_ids.append(tk['input_ids'])
            attention_mask.append(tk['attention_mask'])
            token_type_ids.append(tk['token_type_ids'])

        return pd.DataFrame({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        })

    def mms_fit_scale(self, ds):
        self.mms.fit(ds)
        return self.mms.transform(ds)

    def mms_fit(self, ds):
        self.mms.fit(ds)

    def mms_scale(self, ds):
        return self.mms.transform(ds)

    def _pad_single_doc(self, doc_seq: torch.Tensor) -> torch.Tensor:
        assert len(doc_seq.shape) == 2
        doc_seq = doc_seq[0:self.max_doc_length, 0:self.max_sent_length]
        doc_seq[:, -1] = SEP
        padded_seq = torch.zeros(self.max_doc_length, self.max_sent_length).type(torch.long)
        r = min(self.max_doc_length, doc_seq.shape[0])
        c = min(self.max_sent_length, doc_seq.shape[1])
        padded_seq[0:r, 0:c] = doc_seq[0:r, 0:c]
        return padded_seq

    def pad_docs(self, input_ids: List[torch.Tensor], attention_mask: List[torch.Tensor]) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        padded_inputs_ids = torch.stack([
            self._pad_single_doc(s) for s in input_ids
        ], dim=0)
        padded_attention_mask = torch.stack([
            self._pad_single_doc(a) for a in attention_mask
        ], dim=0)
        length = torch.tensor([min(len(d), self.max_doc_length) for d in input_ids], dtype=torch.int)
        return padded_inputs_ids, padded_attention_mask, length
    
    def cut_and_pad_docs(self, docs: pd.Series):
        logger.debug("cutting documents ...")
        doc_seq = docs.apply(self.cut_doc)

        logger.debug('tokenizing documents ...')
        bert_inputs = self.bert_tokenize(doc_seq)
        input_ids = list(bert_inputs.input_ids)
        attention_mask = list(bert_inputs.attention_mask)

        logger.debug('padding documents ...')
        o = self.pad_docs(input_ids, attention_mask)
        logger.debug('document cut and padded successfully')

        return o
