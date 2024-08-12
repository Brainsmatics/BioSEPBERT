# encoding: utf-8
# @author: xkchai
# @license: (C) Copyright 2020-2024, Node Supply Chain Manager Corporation Limited.
# @contact: chaixiaokang@hust.edu.cn


import os
import json
import tqdm
import numpy as np


class Trie:
    def __init__(self):
        self.children = {}
        self.is_end = False

    def build(self, token):
        root = self
        for c in token:
            if c not in root.children:
                root.children[c] = Trie()
            root = root.children[c]
        root.is_end = True
        return

    def search(self, sentence):
        # search the first entity and return entity position
        root = self
        start, end = -100, -100
        for (i, c) in enumerate(sentence):
            if start != -100 and end != -100:
                return (start, end)
            if c.lower() in root.children:
                start = i
                end = self.match(root, sentence, i)
        return (start, end)

    def match(self, root, sentence, i):
        end = -100
        while i < len(sentence):
            c = sentence[i].lower()
            if c not in root.children:
                break
            if root.children[c].is_end:
                end = i
            root = root.children[c]
            i += 1
        return end


class NERDataProcessor(object):
    def __init__(self, root, is_lower=True, no_entity_label='O', predict=False, cross_validation=None,
                 train_ids=None, dev_ids=None, test_ids=None, Denoising=False):
        self.task_data_dir = root
        self.train_path = os.path.join(self.task_data_dir, 'train.tsv')
        self.dev_path = os.path.join(self.task_data_dir, 'devel.tsv')
        self.test_path = os.path.join(self.task_data_dir, 'test.tsv')
        self.label2id = None
        self.id2label = None
        self._get_labels()
        self.num_labels = len(self.label2id.keys())
        self.no_entity_label = no_entity_label
        self.is_lower = is_lower
        self.cross_validation = cross_validation
        self.cross_data = os.path.join(self.task_data_dir, 'WhiteText.tsv')
        self.train_ids = train_ids
        self.dev_ids = dev_ids
        self.test_ids = test_ids
        self.predict = predict
        self.Denoising = Denoising
        if self.Denoising:
            self.dictionary, self.dictionary_correct, self.entity_s = self._get_dictionary_tree()

    def get_train_sample(self):
        if self.cross_validation:
            return self._pre_cross_process(self.cross_data, self.train_ids)
        elif self.Denoising:
            return self._pre_process_dictionary(self._pre_process(self.train_path))
        else:
            return self._pre_process(self.train_path)

    def get_dev_sample(self):
        if self.cross_validation:
            return self._pre_cross_process(self.cross_data, self.dev_ids)
        elif self.Denoising:
            return self._pre_process_dictionary(self._pre_process(self.dev_path))
        else:
            return self._pre_process(self.dev_path)

    def get_test_sample(self):
        if self.cross_validation:
            return self._pre_cross_process(self.cross_data, self.test_ids)
        elif self.Denoising and not self.predict:
            return self._pre_process_dictionary(self._pre_process(self.test_path))
        else:
            return self._pre_process(self.test_path)

    def _get_labels(self):
        label_list = ['O', 'B-BrainRegion', 'I-BrainRegion', 'X']
        label2id = {v: i for i, v in enumerate(label_list)}
        id2label = {i: v for i, v in enumerate(label_list)}
        self.label2id = label2id
        self.id2label = id2label

    def _pre_process(self, path):
        '''
        Args:
            path:
        Returns: outputs:{'text':[doc1:List[str], doc2:List[str], ...],
                          'label':[doc1:List[str], doc2:List[str], ...]}
        '''
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]

        outputs = {'text': [], 'label': []}
        text, label = [], []
        for line in lines:
            if line:
                text_ = line.split(' ')[0]
                label_ = line.split(' ')[1] if not self.predict else 'O'
                text.append(text_)
                label.append(label_)
            elif text:
                outputs['text'].append(text)
                outputs['label'].append(label)
                text, label = [], []
        '''Augmentation'''
        # for line in lines:
        #     if line:
        #         text_ = line.split(' ')[0]
        #         label_ = line.split(' ')[1] if not self.predict else 'O'
        #         text.append(text_)
        #         label.append(label_)
        #     elif text:
        #         outputs['text'].append(text)
        #         outputs['label'].append(label)
        #         text, label = [], []
        # if 'train.tsv' in path:
        #     with open(self.dev_path, 'r', encoding='utf-8') as f:
        #         lines = f.readlines()
        #     lines = [line.strip() for line in lines]
        #     text, label = [], []
        #     for line in lines:
        #         if line:
        #             text_ = line.split(' ')[0]
        #             label_ = line.split(' ')[1] if not self.predict else 'O'
        #             text.append(text_)
        #             label.append(label_)
        #         elif text:
        #             outputs['text'].append(text)
        #             outputs['label'].append(label)
        #             text, label = [], []
        return outputs

    def _pre_cross_process(self, path, ids):
        '''
        Args:
            path:
        Returns: outputs:{'text':[doc1:List[str], doc2:List[str], ...],
                          'label':[doc1:List[str], doc2:List[str], ...]}
        '''
        with open(path, 'r', encoding='utf-8') as f:
            docs = f.read().strip().split('-----\n')
        outputs = {'text': [], 'label': []}
        docs = [doc.strip().split('\n\n') for doc in docs]
        for id in ids:
            doc = docs[id]
            for lines in doc:
                text = [l.split(' ')[0] for l in lines.split('\n')]
                label = [l.split(' ')[1] if not self.predict else 'O' for l in lines.split('\n')]
                outputs['text'].append(text)
                outputs['label'].append(label)
        return outputs

    def _get_dictionary_tree(self):
        with open('../dataset/Corpus/WhiteText.tsv', 'r', encoding='utf-8') as f:
            data = f.read().strip().split('-----\n')  # [docs1, doc2, ...]
        docs = [i.strip().split('\n\n') for i in data]  # [[sentence1, sentence2, ...], [sentence1, sentence2, ...], ...]
        dictionary = Trie()
        # 1. build dictionary tree
        # text:[[word1, word2, ...], ...]
        text = []
        # label:[[label1, label2, ...], ...]
        label = []
        entity_bio = []
        for doc in docs:
            for sentence in doc:
                text_ = []
                label_ = []
                for line in sentence.strip().split('\n'):
                    entity, entity_label = line.split(' ')
                    text_.append(entity)
                    label_.append(entity_label)
                    if entity_bio and entity_label != 'I-BrainRegion':
                        dictionary.build(entity_bio)
                        entity_bio = []
                    if entity_label != 'O':
                        entity_bio.append(entity.lower())
                text.append(text_)
                label.append(label_)
        # 2. search dictionary tree and count each entity number
        entity_count = {}
        for (i, sentence) in enumerate(text):
            start, end, end_ = -1, -1, -1
            while end_ != -100:
                start_, end_ = dictionary.search(sentence[end + 1:])
                start = start_ + end + 1
                end = end_ + end + 1
                if end_ != -100:
                    entity = ' '.join(sentence[start:end + 1]).lower()
                    if entity not in entity_count:
                        entity_count[entity] = {'total': 1, 'match': 1}
                    else:
                        if label[i][start] == 'B-BrainRegion':
                            entity_count[entity]['match'] += 1
                        entity_count[entity]['total'] += 1
        # 3. select entity which number more than 50% and delete which number less than 25%
        dictionary_correct = Trie()
        entity_select = []
        for k, v in entity_count.items():
            if v['match'] / v['total'] > 0.25:
                entity_select.append(k)
            if v['match'] / v['total'] >= 0.5:
                dictionary_correct.build(k.split())
        return dictionary, dictionary_correct, entity_select

    def _pre_process_dictionary(self, outputs):
        # 1. add label
        for (i, sentence) in enumerate(outputs['text']):
            start, end, end_ = -1, -1, -1
            while end_ != -100:
                start_, end_ = self.dictionary_correct.search(sentence[end + 1:])
                start = start_ + end + 1
                end = end_ + end + 1
                if end_ != -100:
                    if outputs['label'][i][start] == 'O':
                        outputs['label'][i][start] = 'B-BrainRegion'
                        for j in range(start + 1, end + 1):
                            outputs['label'][i][j] = 'I-BrainRegion'

        # 2. delete label
        for (i, sentence) in enumerate(outputs['text']):
            start, end, end_ = -1, -1, -1
            while end_ != -100:
                start_, end_ = self.dictionary.search(sentence[end + 1:])
                start = start_ + end + 1
                end = end_ + end + 1
                if end_ != -100 and ' '.join(sentence[start:end + 1]).lower() not in self.entity_s:
                    outputs['label'][i][start] = 'O'
                    for j in range(start + 1, end + 1):
                        outputs['label'][i][j] = 'O'
        return outputs


