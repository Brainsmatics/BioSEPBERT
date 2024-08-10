import os
import torch
import torch.nn as nn
import numpy as np


class REModel(nn.Module):
    def __init__(self, tokenizer, model, num_labels):
        super(REModel, self).__init__()
        self.model = model
        self.model.resize_token_embeddings(len(tokenizer))
        self.classifier = nn.Linear(in_features=self.model.config.hidden_size*2, out_features=num_labels)
        self.loss_CE = nn.CrossEntropyLoss()
        self.loss_BCE = nn.BCELoss()

    def forward(self, input_ids, token_type_ids, attention_mask, flag, labels=None):
        device = input_ids.device
        outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]   # batch, seq, hidden
        batch_size, seq_len, hidden_size = last_hidden_state.shape
        entity_hidden_state = torch.Tensor(batch_size, 2*hidden_size)  # batch, 2*hidden
        # flag: batch, 2
        for i in range(batch_size):
            sub_start_idx, sub_end_idx, obj_start_idx, obj_end_idx = flag[i, 0], flag[i, 1], flag[i, 2], flag[i, 3]
            sub_start_entity = last_hidden_state[i, sub_start_idx, :].view(hidden_size, )   # s_start: hidden,
            #sub_end_entity = last_hidden_state[i, sub_end_idx, :].view(hidden_size, )   # s_end: hidden,
            obj_start_entity = last_hidden_state[i, obj_start_idx, :].view(hidden_size, )   # o_start: hidden,
            #obj_end_entity = last_hidden_state[i, obj_end_idx, :].view(hidden_size, )   # o_end: hidden,
            #entity_hidden_state[i] = torch.cat([sub_start_entity, sub_end_entity, obj_start_entity, obj_end_entity], dim=-1)
            entity_hidden_state[i] = torch.cat([sub_start_entity, obj_start_entity], dim=-1)
        entity_hidden_state = entity_hidden_state.to(device)
        logits = self.classifier(entity_hidden_state)
        if labels is not None:
            loss = self.loss_CE(logits, labels)
            return loss, logits
        return logits


class REDirectionModel(nn.Module):
    def __init__(self, tokenizer, model, num_labels):
        super(REDirectionModel, self).__init__()
        self.model = model
        self.model.resize_token_embeddings(len(tokenizer))
        self.classifier = nn.Linear(in_features=self.model.config.hidden_size*2, out_features=num_labels)
        self.loss_CE = nn.CrossEntropyLoss()
        self.loss_BCE = nn.BCELoss()

    def forward(self, input_ids, token_type_ids, attention_mask, flag, labels=None):
        device = input_ids.device
        outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]   # batch, seq, hidden
        batch_size, seq_len, hidden_size = last_hidden_state.shape
        entity_hidden_state = torch.Tensor(batch_size, 2*hidden_size)  # batch, 2*hidden
        # flag: batch, 2
        for i in range(batch_size):
            sub_start_idx, sub_end_idx, obj_start_idx, obj_end_idx = flag[i, 0], flag[i, 1], flag[i, 2], flag[i, 3]
            sub_start_entity = last_hidden_state[i, sub_start_idx, :].view(hidden_size, )   # s_start: hidden,
            obj_start_entity = last_hidden_state[i, obj_start_idx, :].view(hidden_size, )   # o_start: hidden,
            entity_hidden_state[i] = torch.cat([sub_start_entity, obj_start_entity], dim=-1)
        entity_hidden_state = entity_hidden_state.to(device)
        logits = self.classifier(entity_hidden_state)
        if labels is not None:
            loss = self.loss_CE(logits, labels)
            return loss, logits
        return logits
