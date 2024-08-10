# encoding: utf-8
# @author: xkchai
# @contact: chaixiaokang@hust.edu.cn


import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Set, Union, Optional


class REDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128,
            model_type='bert',
            ngram_dict=None
    ):
        super(REDataset, self).__init__()
        self.data_processor = data_processor
        self.texts = samples['text']
        self.flags = samples['flag']

        if not self.data_processor.predict:
            self.labels = samples['label']

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.model_type = model_type
        self.ngram_dict = ngram_dict

    def __getitem__(self, idx):
        text, flag = self.texts[idx], self.flags[idx]
        inputs = self.tokenizer.encode_plus(text, max_length=self.max_length, padding='max_length', truncation=True)

        s_start_encode = self.tokenizer.encode(flag[0])
        s_start_idx = self.data_processor.search(inputs['input_ids'], s_start_encode[1:-1])
        s_end_encode = self.tokenizer.encode(flag[1])
        s_end_idx = self.data_processor.search(inputs['input_ids'], s_end_encode[1:-1])

        o_start_encode = self.tokenizer.encode(flag[2])
        o_start_idx = self.data_processor.search(inputs['input_ids'], o_start_encode[1:-1])
        o_end_encode = self.tokenizer.encode(flag[3])
        o_end_idx = self.data_processor.search(inputs['input_ids'], o_end_encode[1:-1])
        if not self.data_processor.predict:
            label = self.labels[idx]
            return torch.tensor(inputs['input_ids']), \
                       torch.tensor(inputs['token_type_ids']), \
                       torch.tensor(inputs['attention_mask']), \
                       torch.tensor([s_start_idx, s_end_idx, o_start_idx, o_end_idx]).long(), \
                       torch.tensor(label).long()
        else:
            return torch.tensor(inputs['input_ids']), \
                       torch.tensor(inputs['token_type_ids']).long(), \
                       torch.tensor(inputs['attention_mask']).float(), \
                       torch.tensor([s_start_idx, s_end_idx, o_start_idx, o_end_idx]).long()

    def __len__(self):
        return len(self.texts)
