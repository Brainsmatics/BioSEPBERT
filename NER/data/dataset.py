import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Set, Union, Optional


class BioSEPDataset(Dataset):
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
        super(BioSEPDataset, self).__init__()
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.ngram_dict = ngram_dict
        self.model_type = model_type
        self.texts = samples['text']
        if not self.data_processor.predict:
            self.labels = samples['label']
        else:
            self.labels = None

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.labels:
            label = self.labels[idx]
            inputs = convert_examples_to_BioSEP_tokens(text, label, self.max_length, self.tokenizer,
                                                     self.data_processor.label2id)
            return inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'], \
                   inputs['label_ids'], inputs['label_mask_ids'], \
                   inputs['B_start_ids'], inputs['B_end_ids'], inputs['I_start_ids'], inputs['I_end_ids']

        else:
            inputs = convert_examples_to_BioSEP_tokens(text, None, self.max_length, self.tokenizer,
                                                     self.data_processor.label2id)
            return inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'],\
                   inputs['label_ids'], inputs['label_mask_ids'], \
                   inputs['B_start_ids'], inputs['B_end_ids'], inputs['I_start_ids'], inputs['I_end_ids']

    def __len__(self):
        return len(self.texts)


def convert_examples_to_BioSEP_tokens(text: List[str], label: Optional[List[str]], max_seq_length=128,
                               tokenizer=None, label2id=None, return_tensors=False):
    inputs = {'input_ids': [], 'attention_mask': [], 'token_type_ids': [],
              'label_ids': [], 'label_mask_ids': [],
              'B_start_ids': [], 'B_end_ids': [], 'I_start_ids': [], 'I_end_ids': []
              }

    text_list = text
    label_list = label
    if label_list:
        assert len(text_list) == len(label_list)

    B, I, O = label2id['B-BrainRegion'], label2id['I-BrainRegion'], label2id['O']
    ignore_label = -1
    tokens = []
    label_tokens = []
    label_masks = []
    for i in range(len(text_list)):
        token = tokenizer.tokenize(text_list[i])
        tokens.extend(token)
        if label_list:
            label_mask = [label2id[label_list[i]]] + [-100] * (len(token)-1)
            if label_list[i] == 'B-BrainRegion':
                label_token = [B] + [ignore_label] * (len(token)-1)
            elif label_list[i] == 'I-BrainRegion':
                label_token = [I] * len(token)
            else:
                label_token = [O] * len(token)
        else:
            label_token = [0] * len(token)
            label_mask = [0] + [-100] * (len(token)-1)
        label_tokens.extend(label_token)
        label_masks.extend(label_mask)

    assert len(tokens) == len(label_tokens)
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        label_tokens = label_tokens[0:(max_seq_length - 2)]
        label_masks = label_masks[0:(max_seq_length - 2)]

    ntokens = []
    nlabels = []
    segment_ids = []
    nlabel_mask = []
    ntokens.append("[CLS]")
    nlabels.append(-100)
    segment_ids.append(0)
    nlabel_mask.append(-100)
    for i in range(len(tokens)):
        ntokens.append(tokens[i])
        nlabels.append(label_tokens[i])
        segment_ids.append(0)
        nlabel_mask.append(label_masks[i])
    ntokens.append("[SEP]")
    nlabels.append(-100)
    segment_ids.append(0)
    nlabel_mask.append(-100)
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    label_ids = nlabels


    B_start = [-100]
    B_end = [-100]
    I_start = [-100]
    I_end = [-100]
    for i in range(1, len(label_ids)-1):
        if label_ids[i] == B:
            B_start.append(1)
            if label_ids[i+1] != ignore_label:
                B_end.append(1)
            else:
                B_end.append(0)
            I_start.append(0)
            I_end.append(0)
        elif label_ids[i] == ignore_label and label_ids[i+1] != ignore_label:
            B_start.append(0)
            B_end.append(1)
            I_start.append(0)
            I_end.append(0)
        elif label_ids[i] == I and label_ids[i-1] != I:
            B_start.append(0)
            B_end.append(0)
            I_start.append(1)
            if label_ids[i+1] != I:
                I_end.append(1)
            else:
                I_end.append(0)
        elif label_ids[i] == I and label_ids[i+1] != I:
            B_start.append(0)
            B_end.append(0)
            I_start.append(0)
            I_end.append(1)
        else:
            B_start.append(0)
            B_end.append(0)
            I_start.append(0)
            I_end.append(0)
    B_start += [-100]
    B_end += [-100]
    I_start += [-100]
    I_end += [-100]
    label_ids = [2 if i == -1 else i for i in label_ids]

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        segment_ids.append(0)
        input_mask.append(0)
        label_ids.append(-100)
        nlabel_mask.append(-100)
        B_start.append(-100)
        B_end.append(-100)
        I_start.append(-100)
        I_end.append(-100)

    assert len(input_ids) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(B_start) == max_seq_length
    assert len(I_start) == max_seq_length
    assert len(nlabel_mask) == max_seq_length

    inputs['input_ids'] = np.array(input_ids)
    inputs['token_type_ids'] = np.array(segment_ids)
    inputs['attention_mask'] = np.array(input_mask)
    inputs['label_ids'] = np.array(label_ids)
    inputs['B_start_ids'] = np.array(B_start)
    inputs['B_end_ids'] = np.array(B_end)
    inputs['I_start_ids'] = np.array(I_start)
    inputs['I_end_ids'] = np.array(I_end)
    inputs['label_mask_ids'] = np.array(nlabel_mask)

    if return_tensors:
        inputs['input_ids'] = torch.tensor(inputs['input_ids'])
        inputs['token_type_ids'] = torch.tensor(inputs['token_type_ids'])
        inputs['attention_mask'] = torch.tensor(inputs['attention_mask'])
        inputs['label_ids'] = torch.tensor(inputs['label_ids'])
        inputs['B_start_ids'] = torch.tensor(inputs['B_start_ids'])
        inputs['B_end_ids'] = torch.tensor(inputs['B_end_ids'])
        inputs['I_start_ids'] = torch.tensor(inputs['I_start_ids'])
        inputs['I_end_ids'] = torch.tensor(inputs['I_end_ids'])
        inputs['label_mask_ids'] = torch.tensor(inputs['label_mask_ids'])

    return inputs

