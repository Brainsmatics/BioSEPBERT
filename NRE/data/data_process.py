import os
import json
import tqdm
import numpy as np
import random


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
        # sentence: [word1, word2, ...]
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


class REDataProcessor(object):
    def __init__(self, root, repre=False):
        self.task_data_dir = root
        self.train_path = os.path.join(self.task_data_dir, 'train.json')
        self.dev_path = os.path.join(self.task_data_dir, 'devel.json')
        self.test_path = os.path.join(self.task_data_dir, 'test.json')
        self.num_labels = 2
        self.predict = False
        self.repre = repre
        self.dictionary, self.dictionary_correct, self.entity_s = self._get_dictionary_tree()

    def get_train_sample(self):
        if self.repre:
            return self._repre_process(self.train_path)
        else:
            return self._pre_process(self.train_path)

    def get_dev_sample(self):
        return self._pre_process(self.dev_path)

    def get_test_sample(self):
        return self._pre_process(self.test_path)

    def _pre_process(self, path):
        with open(path, 'r', encoding='utf8') as f:
            data = json.load(f)
        result = {'text': [], 'label': [], 'flag': []}
        for k, v in data.items():
            for i in range(len(v['text'])):
                text = v['text'][i]
                sentences = v['sentences'][i]
                for sentence in sentences:
                    start1, end1, label, start2, end2 = sentence
                    if start1 > start2:
                        start1, start2 = start2, start1
                        end1, end2 = end2, end1
                    text1 = text[:start1] + '<s> ' + text[start1:end1+1] + ' </s>' + text[end1+1:start2] +\
                           '<o> ' + text[start2:end2+1] + ' </o>' + text[end2+1:]
                    result['text'].append(text1)
                    result['label'].append(1 if label == 'True' else 0)
                    result['flag'].append(('<s>', '</s>', '<o>', '</o>'))
        return result

    def _repre_process(self, path):
        with open(path, 'r', encoding='utf8') as f:
            data = json.load(f)
        result = {'text': [], 'label': [], 'flag': []}
        total_num = 0
        for k, v in data.items():
            for i in range(len(v['text'])):
                text = v['text'][i]
                sentences = v['sentences'][i]
                for sentence in sentences:
                    start1, end1, label, start2, end2 = sentence
                    if start1 > start2:
                        start1, start2 = start2, start1
                        end1, end2 = end2, end1
                    text1 = text[:start1] + '<s> ' + text[start1:end1+1] + ' </s>' + text[end1+1:start2] +\
                           '<o> ' + text[start2:end2+1] + ' </o>' + text[end2+1:]
                    result['text'].append(text1)
                    result['label'].append(1 if label == 'True' else 0)
                    result['flag'].append(('<s>', '</s>', '<o>', '</o>'))
                    if label == 'True':
                        result['text'].append(text1)
                        result['label'].append(1 if label == 'True' else 0)
                        result['flag'].append(('<s>', '</s>', '<o>', '</o>'))

                    sub_set, obj_set = [], []  # [[start1, end1], [start2, end2], ...]
                    sub, obj = text[start1: end1+1].split(), text[start2: end2+1].split()  # [word1, word2, ...]
                    sub_end, obj_end = len(sub)-1, len(obj)-1

                    while sub_end != -100:
                        sub_start, sub_end = self.dictionary_correct.search(sub[:sub_end])
                        if sub_end != -100:
                            total_num += 1
                            sub_start_ = start1 + len(' '.join(sub[:sub_start]))+1 if sub_start else start1
                            sub_end_ = start1 + len(' '.join(sub[:sub_end+1]))-1
                            sub_set.append([sub_start_, sub_end_])
                            text1 = text[:sub_start_] + '<s> ' + text[sub_start_:sub_end_ + 1] \
                                    + ' </s>' + text[sub_end_ + 1:start2] + '<o> '\
                                    + text[start2:end2 + 1] + ' </o>' + text[end2 + 1:]
                            result['text'].append(text1)
                            result['label'].append(1 if label == 'True' else 0)
                            result['flag'].append(('<s>', '</s>', '<o>', '</o>'))
                            if label == 'True':
                                result['text'].append(text1)
                                result['label'].append(1 if label == 'True' else 0)
                                result['flag'].append(('<s>', '</s>', '<o>', '</o>'))
                    while obj_end != -100:
                        obj_start, obj_end = self.dictionary_correct.search(obj[:obj_end])
                        if obj_end != -100:
                            total_num += 1
                            obj_start_ = start2 + len(' '.join(obj[:obj_start])) + 1 if obj_start else start2
                            obj_end_ = start2 + len(' '.join(obj[:obj_end + 1])) - 1
                            obj_set.append([obj_start_, obj_end_])
                            text1 = text[:start1] + '<s> ' + text[start1:end1 + 1] + ' </s>' \
                                    + text[end1 + 1:obj_start_] + '<o> ' + text[obj_start_:obj_end_ + 1] \
                                    + ' </o>' + text[obj_end_ + 1:]
                            result['text'].append(text1)
                            result['label'].append(1 if label == 'True' else 0)
                            result['flag'].append(('<s>', '</s>', '<o>', '</o>'))
                            if label == 'True':
                                result['text'].append(text1)
                                result['label'].append(1 if label == 'True' else 0)
                                result['flag'].append(('<s>', '</s>', '<o>', '</o>'))
        return result

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
        # 3. select entity which number more than 25% and delete which number less than 50%
        dictionary_correct = Trie()
        entity_select = []
        for k, v in entity_count.items():
            if v['match'] / v['total'] > 0.25:
                entity_select.append(k)
            if v['match'] / v['total'] >= 0.5:
                dictionary_correct.build(k.split())
        return dictionary, dictionary_correct, entity_select

    def search(self, sequence, pattern):
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return 0

    def build_text(self, data):
        text = data['text']
        result = []
        outputs = {'text': [], 'flag': [], "spo_list": []}
        for sub in data['sub_list']:
            for obj in data['obj_list']:
                if sub == obj:
                    continue
                sub_flag = ['<s>', '</s>']
                obj_flag = ['<o>', '</o>']
                sub_start = self.search(text, sub)
                sub_end = sub_start + len(sub)
                text2 = text[:sub_start] + sub_flag[0] + sub + sub_flag[1] + text[sub_end:]
                obj_start = self.search(text2, obj)
                obj_end = obj_start + len(obj)
                text3 = text2[:obj_start] + obj_flag[0] + obj + obj_flag[1] + text2[obj_end:]
                result.append(
                    {'text': text3, 'flag': (sub_flag[0], obj_flag[0]), 'spo_list': {'subject': sub, 'object': obj}})
                outputs['text'].append(text3)
                outputs['flag'].append((sub_flag[0], obj_flag[0]))
                outputs['spo_list'].append({'subject': sub, 'object': obj})
        return result, outputs

    def get_train_sample_bio(self):
        if self.repre:
            return self._repre_process_bio(self.train_path)
        else:
            return self._pre_process_bio(self.train_path)

    def get_dev_sample_bio(self):
        return self._pre_process_bio(self.dev_path)

    def get_test_sample_bio(self):
        return self._pre_process_bio(self.test_path)

    def _pre_process_bio(self, path):
        with open(path, 'r', encoding='utf8') as f:
            data = json.load(f)
        result = {'text': [], 'label': []}
        for k, v in data.items():
            for i in range(len(v['text'])):
                text = v['text'][i]
                sentences = v['sentences'][i]
                for sentence in sentences:
                    start1, end1, label, start2, end2 = sentence
                    if start1 > start2:
                        start1, start2 = start2, start1
                        end1, end2 = end2, end1
                    text1 = text[:start1] + '@BrainRegion$' + text[end1+1:start2] + '@BrainRegion$' + text[end2+1:]
                    result['text'].append(text1)
                    result['label'].append(1 if label == 'True' else 0)
        return result

    def _repre_process_bio(self, path):
        with open(path, 'r', encoding='utf8') as f:
            data = json.load(f)
        result = {'text': [], 'label': []}
        total_num = 0
        for k, v in data.items():
            for i in range(len(v['text'])):
                text = v['text'][i]
                sentences = v['sentences'][i]
                for sentence in sentences:
                    start1, end1, label, start2, end2 = sentence
                    if start1 > start2:
                        start1, start2 = start2, start1
                        end1, end2 = end2, end1
                    text1 = text[:start1] + '@BrainRegion$' + text[end1+1:start2] + '@BrainRegion$' + text[end2+1:]
                    result['text'].append(text1)
                    result['label'].append(1 if label == 'True' else 0)
                    # repeat add correct case one time
                    if label == 'True':
                        result['text'].append(text1)
                        result['label'].append(1 if label == 'True' else 0)

                    sub_set, obj_set = [], []  # [[start1, end1], [start2, end2], ...]
                    sub, obj = text[start1: end1+1].split(), text[start2: end2+1].split()  # [word1, word2, ...]
                    sub_end, obj_end = len(sub)-1, len(obj)-1

                    while sub_end != -100:
                        sub_start, sub_end = self.dictionary_correct.search(sub[:sub_end])
                        if sub_end != -100:
                            total_num += 1
                            sub_start_ = start1 + len(' '.join(sub[:sub_start]))+1 if sub_start else start1
                            sub_end_ = start1 + len(' '.join(sub[:sub_end+1]))-1
                            sub_set.append([sub_start_, sub_end_])
                            text1 = text[:sub_start_] + '@BrainRegion$' + text[sub_end_ + 1:start2] + \
                                    '@BrainRegion$' + text[end2 + 1:]
                            result['text'].append(text1)
                            result['label'].append(1 if label == 'True' else 0)
                            if label == 'True':
                                result['text'].append(text1)
                                result['label'].append(1 if label == 'True' else 0)
                    while obj_end != -100:
                        obj_start, obj_end = self.dictionary_correct.search(obj[:obj_end])
                        if obj_end != -100:
                            total_num += 1
                            obj_start_ = start2 + len(' '.join(obj[:obj_start])) + 1 if obj_start else start2
                            obj_end_ = start2 + len(' '.join(obj[:obj_end + 1])) - 1
                            obj_set.append([obj_start_, obj_end_])
                            text1 = text[:start1] + '@BrainRegion$' + text[end1 + 1:obj_start_] + \
                                    '@BrainRegion$' + text[obj_end_ + 1:]
                            result['text'].append(text1)
                            result['label'].append(1 if label == 'True' else 0)
                            if label == 'True':
                                result['text'].append(text1)
                                result['label'].append(1 if label == 'True' else 0)
        return result


class RESLDataProcessor(object):
    def __init__(self, root, repre=False):
        self.task_data_dir = root
        self.train_path = os.path.join(self.task_data_dir, 'train.json')
        self.dev_path = os.path.join(self.task_data_dir, 'devel.json')
        self.test_path = os.path.join(self.task_data_dir, 'test.json')
        self.num_labels = 2
        self.predict = False
        self.repre = repre
        self.dictionary, self.dictionary_correct, self.entity_s = self._get_dictionary_tree()

    def get_train_sample(self):
        if self.repre:
            return self._repre_process(self.train_path)
        else:
            return self._pre_process(self.train_path)

    def get_dev_sample(self):
        return self._pre_process(self.dev_path)

    def get_test_sample(self):
        return self._pre_process(self.test_path)

    def _pre_process(self, path):
        with open(path, 'r', encoding='utf8') as f:
            data = json.load(f)
        result = {'text': [], 'label': [], 'flag': []}
        for i in range(len(data['text'])):
            text = data['text'][i]
            sentences = data['sen'][i]
            start1, end1, label, start2, end2 = sentences
            if start1 > start2:
                start1, start2 = start2, start1
                end1, end2 = end2, end1
            text1 = text[:start1] + '<s> ' + text[start1:end1+1] + ' </s>' + text[end1+1:start2] +\
                   '<o> ' + text[start2:end2+1] + ' </o>' + text[end2+1:]
            result['text'].append(text1)
            result['label'].append(1 if label == 'True' else 0)
            result['flag'].append(('<s>', '</s>', '<o>', '</o>'))
        return result

    def _repre_process(self, path):
        with open(path, 'r', encoding='utf8') as f:
            data = json.load(f)
        result = {'text': [], 'label': [], 'flag': []}
        total_num = 0
        for i in range(len(data['text'])):
            text = data['text'][i]
            sentences = data['sen'][i]
            start1, end1, label, start2, end2 = sentences
            if start1 > start2:
                start1, start2 = start2, start1
                end1, end2 = end2, end1
            text1 = text[:start1] + '<s> ' + text[start1:end1+1] + ' </s>' + text[end1+1:start2] +\
                   '<o> ' + text[start2:end2+1] + ' </o>' + text[end2+1:]
            result['text'].append(text1)
            result['label'].append(1 if label == 'True' else 0)
            result['flag'].append(('<s>', '</s>', '<o>', '</o>'))
            if label == 'True':
                result['text'].append(text1)
                result['label'].append(1 if label == 'True' else 0)
                result['flag'].append(('<s>', '</s>', '<o>', '</o>'))

            sub_set, obj_set = [], []  # [[start1, end1], [start2, end2], ...]
            sub, obj = text[start1: end1+1].split(), text[start2: end2+1].split()  # [word1, word2, ...]
            sub_end, obj_end = len(sub)-1, len(obj)-1

            while sub_end != -100:
                sub_start, sub_end = self.dictionary_correct.search(sub[:sub_end])
                if sub_end != -100:
                    total_num += 1
                    sub_start_ = start1 + len(' '.join(sub[:sub_start]))+1 if sub_start else start1
                    sub_end_ = start1 + len(' '.join(sub[:sub_end+1]))-1
                    sub_set.append([sub_start_, sub_end_])
                    text1 = text[:sub_start_] + '<s> ' + text[sub_start_:sub_end_ + 1] \
                            + ' </s>' + text[sub_end_ + 1:start2] + '<o> '\
                            + text[start2:end2 + 1] + ' </o>' + text[end2 + 1:]
                    result['text'].append(text1)
                    result['label'].append(1 if label == 'True' else 0)
                    result['flag'].append(('<s>', '</s>', '<o>', '</o>'))
                    if label == 'True':
                        result['text'].append(text1)
                        result['label'].append(1 if label == 'True' else 0)
                        result['flag'].append(('<s>', '</s>', '<o>', '</o>'))
            while obj_end != -100:
                obj_start, obj_end = self.dictionary_correct.search(obj[:obj_end])
                if obj_end != -100:
                    total_num += 1
                    obj_start_ = start2 + len(' '.join(obj[:obj_start])) + 1 if obj_start else start2
                    obj_end_ = start2 + len(' '.join(obj[:obj_end + 1])) - 1
                    obj_set.append([obj_start_, obj_end_])
                    text1 = text[:start1] + '<s> ' + text[start1:end1 + 1] + ' </s>' \
                            + text[end1 + 1:obj_start_] + '<o> ' + text[obj_start_:obj_end_ + 1] \
                            + ' </o>' + text[obj_end_ + 1:]
                    result['text'].append(text1)
                    result['label'].append(1 if label == 'True' else 0)
                    result['flag'].append(('<s>', '</s>', '<o>', '</o>'))
                    if label == 'True':
                        result['text'].append(text1)
                        result['label'].append(1 if label == 'True' else 0)
                        result['flag'].append(('<s>', '</s>', '<o>', '</o>'))
        return result

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
        # 3. select entity which number more than 25% and delete which number less than 50%
        dictionary_correct = Trie()
        entity_select = []
        for k, v in entity_count.items():
            if v['match'] / v['total'] > 0.25:
                entity_select.append(k)
            if v['match'] / v['total'] >= 0.5:
                dictionary_correct.build(k.split())
        return dictionary, dictionary_correct, entity_select

    def search(self, sequence, pattern):
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return 0

    def build_text(self, data):
        text = data['text']
        result = []
        outputs = {'text': [], 'flag': [], "spo_list": []}
        for sub in data['sub_list']:
            for obj in data['obj_list']:
                if sub == obj:
                    continue
                sub_flag = ['<s>', '</s>']
                obj_flag = ['<o>', '</o>']
                sub_start = self.search(text, sub)
                sub_end = sub_start + len(sub)
                text2 = text[:sub_start] + sub_flag[0] + sub + sub_flag[1] + text[sub_end:]
                obj_start = self.search(text2, obj)
                obj_end = obj_start + len(obj)
                text3 = text2[:obj_start] + obj_flag[0] + obj + obj_flag[1] + text2[obj_end:]
                result.append(
                    {'text': text3, 'flag': (sub_flag[0], obj_flag[0]), 'spo_list': {'subject': sub, 'object': obj}})
                outputs['text'].append(text3)
                outputs['flag'].append((sub_flag[0], obj_flag[0]))
                outputs['spo_list'].append({'subject': sub, 'object': obj})
        return result, outputs

    def get_train_sample_bio(self):
        if self.repre:
            return self._repre_process_bio(self.train_path)
        else:
            return self._pre_process_bio(self.train_path)

    def get_dev_sample_bio(self):
        return self._pre_process_bio(self.dev_path)

    def get_test_sample_bio(self):
        return self._pre_process_bio(self.test_path)

    def _pre_process_bio(self, path):
        with open(path, 'r', encoding='utf8') as f:
            data = json.load(f)
        result = {'text': [], 'label': []}
        for i in range(len(data['text'])):
            text = data['text'][i]
            sentences = data['sen'][i]
            start1, end1, label, start2, end2 = sentences
            if start1 > start2:
                start1, start2 = start2, start1
                end1, end2 = end2, end1
            text1 = text[:start1] + '@BrainRegion$' + text[end1+1:start2] + '@BrainRegion$' + text[end2+1:]
            result['text'].append(text1)
            result['label'].append(1 if label == 'True' else 0)
        return result

    def _repre_process_bio(self, path):
        with open(path, 'r', encoding='utf8') as f:
            data = json.load(f)
        result = {'text': [], 'label': []}
        total_num = 0

        for i in range(len(data['text'])):
            text = data['text'][i]
            sentences = data['sen'][i]
            start1, end1, label, start2, end2 = sentences
            if start1 > start2:
                start1, start2 = start2, start1
                end1, end2 = end2, end1
            text1 = text[:start1] + '@BrainRegion$' + text[end1+1:start2] + '@BrainRegion$' + text[end2+1:]
            result['text'].append(text1)
            result['label'].append(1 if label == 'True' else 0)
            # repeat add correct case one time
            if label == 'True':
                result['text'].append(text1)
                result['label'].append(1 if label == 'True' else 0)

            sub_set, obj_set = [], []  # [[start1, end1], [start2, end2], ...]
            sub, obj = text[start1: end1+1].split(), text[start2: end2+1].split()  # [word1, word2, ...]
            sub_end, obj_end = len(sub)-1, len(obj)-1

            while sub_end != -100:
                sub_start, sub_end = self.dictionary_correct.search(sub[:sub_end])
                if sub_end != -100:
                    total_num += 1
                    sub_start_ = start1 + len(' '.join(sub[:sub_start]))+1 if sub_start else start1
                    sub_end_ = start1 + len(' '.join(sub[:sub_end+1]))-1
                    sub_set.append([sub_start_, sub_end_])
                    text1 = text[:sub_start_] + '@BrainRegion$' + text[sub_end_ + 1:start2] + \
                            '@BrainRegion$' + text[end2 + 1:]
                    result['text'].append(text1)
                    result['label'].append(1 if label == 'True' else 0)
                    if label == 'True':
                        result['text'].append(text1)
                        result['label'].append(1 if label == 'True' else 0)
            while obj_end != -100:
                obj_start, obj_end = self.dictionary_correct.search(obj[:obj_end])
                if obj_end != -100:
                    total_num += 1
                    obj_start_ = start2 + len(' '.join(obj[:obj_start])) + 1 if obj_start else start2
                    obj_end_ = start2 + len(' '.join(obj[:obj_end + 1])) - 1
                    obj_set.append([obj_start_, obj_end_])
                    text1 = text[:start1] + '@BrainRegion$' + text[end1 + 1:obj_start_] + \
                            '@BrainRegion$' + text[obj_end_ + 1:]
                    result['text'].append(text1)
                    result['label'].append(1 if label == 'True' else 0)
                    if label == 'True':
                        result['text'].append(text1)
                        result['label'].append(1 if label == 'True' else 0)
        return result


class RESLAUGDirectionDataProcessor(object):
    def __init__(self, root, repre=False):
        self.task_data_dir = root
        self.train_path = os.path.join(self.task_data_dir, 'train.json')
        self.dev_path = os.path.join(self.task_data_dir, 'devel.json')
        self.test_path = os.path.join(self.task_data_dir, 'test.json')
        self.num_labels = 4
        self.predict = False
        self.repre = repre
        self.dictionary, self.dictionary_correct, self.entity_s = self._get_dictionary_tree()

    def get_train_sample(self):
        if self.repre:
            return self._repre_process(self.train_path)
        else:
            return self._pre_process(self.train_path)

    def get_dev_sample(self):
        return self._pre_process(self.dev_path)

    def get_test_sample(self):
        return self._pre_process(self.test_path)

    def _pre_process(self, path):
        with open(path, 'r', encoding='utf8') as f:
            data = json.load(f)
        result = {'text': [], 'label': [], 'flag': []}
        for i in range(len(data['text'])):
            text = data['text'][i]
            sentences = data['sen_dir'][i]
            start1, end1, label, start2, end2 = sentences
            if start1 > start2:
                start1, start2 = start2, start1
                end1, end2 = end2, end1
            text1 = text[:start1] + '<s> ' + text[start1:end1 + 1] + ' </s>' + text[end1 + 1:start2] + \
                    '<o> ' + text[start2:end2 + 1] + ' </o>' + text[end2 + 1:]
            result['text'].append(text1)
            result['label'].append(label)
            result['flag'].append(('<s>', '</s>', '<o>', '</o>'))
        return result

    def _repre_process(self, path):
        with open(path, 'r', encoding='utf8') as f:
            data = json.load(f)
        result = {'text': [], 'label': [], 'flag': []}
        total_num = 0
        for i in range(len(data['text'])):
            text = data['text'][i]
            sentences = data['sen_dir'][i]
            start1, end1, label, start2, end2 = sentences
            if start1 > start2:
                start1, start2 = start2, start1
                end1, end2 = end2, end1
            text1 = text[:start1] + '<s> ' + text[start1:end1 + 1] + ' </s>' + text[end1 + 1:start2] + \
                    '<o> ' + text[start2:end2 + 1] + ' </o>' + text[end2 + 1:]
            result['text'].append(text1)
            result['label'].append(label)
            result['flag'].append(('<s>', '</s>', '<o>', '</o>'))
            if label:
                result['text'].append(text1)
                result['label'].append(label)
                result['flag'].append(('<s>', '</s>', '<o>', '</o>'))

            sub_set, obj_set = [], []  # [[start1, end1], [start2, end2], ...]
            sub, obj = text[start1: end1 + 1].split(), text[start2: end2 + 1].split()  # [word1, word2, ...]
            sub_end, obj_end = len(sub) - 1, len(obj) - 1

            while sub_end != -100:
                sub_start, sub_end = self.dictionary_correct.search(sub[:sub_end])
                if sub_end != -100:
                    total_num += 1
                    sub_start_ = start1 + len(' '.join(sub[:sub_start])) + 1 if sub_start else start1
                    sub_end_ = start1 + len(' '.join(sub[:sub_end + 1])) - 1
                    sub_set.append([sub_start_, sub_end_])
                    text1 = text[:sub_start_] + '<s> ' + text[sub_start_:sub_end_ + 1] \
                            + ' </s>' + text[sub_end_ + 1:start2] + '<o> ' \
                            + text[start2:end2 + 1] + ' </o>' + text[end2 + 1:]
                    result['text'].append(text1)
                    result['label'].append(label)
                    result['flag'].append(('<s>', '</s>', '<o>', '</o>'))
                    if label:
                        result['text'].append(text1)
                        result['label'].append(label)
                        result['flag'].append(('<s>', '</s>', '<o>', '</o>'))
            while obj_end != -100:
                obj_start, obj_end = self.dictionary_correct.search(obj[:obj_end])
                if obj_end != -100:
                    total_num += 1
                    obj_start_ = start2 + len(' '.join(obj[:obj_start])) + 1 if obj_start else start2
                    obj_end_ = start2 + len(' '.join(obj[:obj_end + 1])) - 1
                    obj_set.append([obj_start_, obj_end_])
                    text1 = text[:start1] + '<s> ' + text[start1:end1 + 1] + ' </s>' \
                            + text[end1 + 1:obj_start_] + '<o> ' + text[obj_start_:obj_end_ + 1] \
                            + ' </o>' + text[obj_end_ + 1:]
                    result['text'].append(text1)
                    result['label'].append(label)
                    result['flag'].append(('<s>', '</s>', '<o>', '</o>'))
                    if label:
                        result['text'].append(text1)
                        result['label'].append(label)
                        result['flag'].append(('<s>', '</s>', '<o>', '</o>'))
        return result

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
        # 3. select entity which number more than 25% and delete which number less than 50%
        dictionary_correct = Trie()
        entity_select = []
        for k, v in entity_count.items():
            if v['match'] / v['total'] > 0.25:
                entity_select.append(k)
            if v['match'] / v['total'] >= 0.5:
                dictionary_correct.build(k.split())
        return dictionary, dictionary_correct, entity_select

    def search(self, sequence, pattern):
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return 0

    def build_text(self, data):
        text = data['text']
        result = []
        outputs = {'text': [], 'flag': [], "spo_list": []}
        for sub in data['sub_list']:
            for obj in data['obj_list']:
                if sub == obj:
                    continue
                sub_flag = ['<s>', '</s>']
                obj_flag = ['<o>', '</o>']
                sub_start = self.search(text, sub)
                sub_end = sub_start + len(sub)
                text2 = text[:sub_start] + sub_flag[0] + sub + sub_flag[1] + text[sub_end:]
                obj_start = self.search(text2, obj)
                obj_end = obj_start + len(obj)
                text3 = text2[:obj_start] + obj_flag[0] + obj + obj_flag[1] + text2[obj_end:]
                result.append(
                    {'text': text3, 'flag': (sub_flag[0], obj_flag[0]), 'spo_list': {'subject': sub, 'object': obj}})
                outputs['text'].append(text3)
                outputs['flag'].append((sub_flag[0], obj_flag[0]))
                outputs['spo_list'].append({'subject': sub, 'object': obj})
        return result, outputs

    def get_train_sample_bio(self):
        if self.repre:
            return self._repre_process_bio(self.train_path)
        else:
            return self._pre_process_bio(self.train_path)

    def get_dev_sample_bio(self):
        return self._pre_process_bio(self.dev_path)

    def get_test_sample_bio(self):
        return self._pre_process_bio(self.test_path)

    def _pre_process_bio(self, path):
        with open(path, 'r', encoding='utf8') as f:
            data = json.load(f)
        result = {'text': [], 'label': []}
        for i in range(len(data['text'])):
            text = data['text'][i]
            sentences = data['sen_dir'][i]
            start1, end1, label, start2, end2 = sentences
            if start1 > start2:
                start1, start2 = start2, start1
                end1, end2 = end2, end1
            text1 = text[:start1] + '@BrainRegion$' + text[end1+1:start2] + '@BrainRegion$' + text[end2+1:]
            result['text'].append(text1)
            result['label'].append(label)
        return result

    def _repre_process_bio(self, path):
        with open(path, 'r', encoding='utf8') as f:
            data = json.load(f)
        result = {'text': [], 'label': []}
        total_num = 0

        for i in range(len(data['text'])):
            text = data['text'][i]
            sentences = data['sen_dir'][i]
            start1, end1, label, start2, end2 = sentences
            if start1 > start2:
                start1, start2 = start2, start1
                end1, end2 = end2, end1
            text1 = text[:start1] + '@BrainRegion$' + text[end1+1:start2] + '@BrainRegion$' + text[end2+1:]
            result['text'].append(text1)
            result['label'].append(label)
            # repeat add correct case one time
            if label:
                result['text'].append(text1)
                result['label'].append(label)

            sub_set, obj_set = [], []  # [[start1, end1], [start2, end2], ...]
            sub, obj = text[start1: end1+1].split(), text[start2: end2+1].split()  # [word1, word2, ...]
            sub_end, obj_end = len(sub)-1, len(obj)-1

            while sub_end != -100:
                sub_start, sub_end = self.dictionary_correct.search(sub[:sub_end])
                if sub_end != -100:
                    total_num += 1
                    sub_start_ = start1 + len(' '.join(sub[:sub_start]))+1 if sub_start else start1
                    sub_end_ = start1 + len(' '.join(sub[:sub_end+1]))-1
                    sub_set.append([sub_start_, sub_end_])
                    text1 = text[:sub_start_] + '@BrainRegion$' + text[sub_end_ + 1:start2] + \
                            '@BrainRegion$' + text[end2 + 1:]
                    result['text'].append(text1)
                    result['label'].append(label)
                    if label:
                        result['text'].append(text1)
                        result['label'].append(label)
            while obj_end != -100:
                obj_start, obj_end = self.dictionary_correct.search(obj[:obj_end])
                if obj_end != -100:
                    total_num += 1
                    obj_start_ = start2 + len(' '.join(obj[:obj_start])) + 1 if obj_start else start2
                    obj_end_ = start2 + len(' '.join(obj[:obj_end + 1])) - 1
                    obj_set.append([obj_start_, obj_end_])
                    text1 = text[:start1] + '@BrainRegion$' + text[end1 + 1:obj_start_] + \
                            '@BrainRegion$' + text[obj_end_ + 1:]
                    result['text'].append(text1)
                    result['label'].append(label)
                    if label:
                        result['text'].append(text1)
                        result['label'].append(label)
        return result
