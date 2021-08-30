from transformers import BertTokenizer
import re
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Preprocessor:
    def __init__(
            self,
            bert_tokenizer_version,
            max_sent_length: int = 512,
            truncation: bool = True,
    ):
        self.tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_version)
        self.max_length = max_sent_length
        self.truncation = truncation

    def sep_label_unlabel(self, labels, unlabeled_mark, *dfs):
        labeled = [], unlabeled = []
        for df in dfs:
            labeled.append(df[labels != unlabeled_mark])
            unlabeled.append(df[labels == unlabeled_mark])
        labeled_labels = labels[labels != unlabeled_mark]
        return labeled_labels, tuple(labeled), tuple(unlabeled)

    def get_label_dict(self, labels, unlabeled_mark=None) -> dict:
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
        para = re.sub('\s*([。！？;\?])([^”’])\s*', r"\1\n\2", para)  # 单字符断句符
        para = re.sub('\s*(\.{6})([^”’])\s*', r"\1\n\2", para)  # 英文省略号
        para = re.sub('\s*(\…{2})([^”’])\s*', r"\1\n\2", para)  # 中文省略号
        para = re.sub('\s*([。！？\?][”’])([^，。！？\?])\s*', r'\1\n\2', para)
        para = re.sub(r'\s+', r'\n', para)
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        return para.split("\n")

    def bert_tokenize(self, doc_seq) -> pd.DataFrame:
        input_ids = []
        attention_mask = []
        token_type_ids = []
        for d in doc_seq:
            tk = self.tokenizer(
                d, return_tensors='pt', padding=True,
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

    def min_max_scale(self, ds):
        mms = MinMaxScaler()
        return mms.fit_transform(ds)
