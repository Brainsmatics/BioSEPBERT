import os
import torch
import torch.nn as nn
import numpy as np


class BERT_SEPModel(nn.Module):
    def __init__(self, encoder):
        super(BERT_SEPModel, self).__init__()
        self.hidden_size = 768
        self.features_size = 64
        self.num_classes = 3
        self.model = encoder
        self.bert_layer = nn.Linear(in_features=self.hidden_size, out_features=self.num_classes)
        self.start_layer1 = nn.Linear(in_features=self.hidden_size, out_features=1)
        self.start_layer2 = nn.Linear(in_features=self.hidden_size, out_features=1)
        self.end_layer1 = nn.Linear(in_features=self.hidden_size, out_features=1)
        self.end_layer2 = nn.Linear(in_features=self.hidden_size, out_features=1)
        self.loss_CE = nn.CrossEntropyLoss()
        self.loss_BCE = nn.BCELoss()

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, label_mask=None):
        outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids,
                             attention_mask=attention_mask)
        logits = outputs.get('last_hidden_state')
        # logits = outputs.get('logits')
        bert_layer = self.bert_layer(logits)
        B_start = self.start_layer1(logits).sigmoid().squeeze(-1)
        B_end = self.end_layer1(logits).sigmoid().squeeze(-1)
        I_start = self.start_layer2(logits).sigmoid().squeeze(-1)
        I_end = self.end_layer2(logits).sigmoid().squeeze(-1)
        loss = None
        if label_mask is not None:
            active = label_mask != -100
            loss_BS = self.loss_BCE(B_start[active].float(), labels[1][active].float())
            loss_BE = self.loss_BCE(B_end[active].float(), labels[2][active].float())
            loss_IS = self.loss_BCE(I_start[active].float(), labels[3][active].float())
            loss_IE = self.loss_BCE(I_end[active].float(), labels[4][active].float())
            loss_bert = self.loss_CE(bert_layer.view(-1, self.num_classes)[active.view(-1)],
                                     labels[0].view(-1)[active.view(-1)])
            loss = 1*(loss_BS + loss_BE + loss_IS + loss_IE) + loss_bert
        return loss, bert_layer, B_start, B_end, I_start, I_end