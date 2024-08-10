# encoding: utf-8
# @author: xkchai
# @contact: chaixiaokang@hust.edu.cn


import os
import json
import torch
import numpy as np
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from NER.utils import ProgressBar
from NER.conlleval import conll_eval
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter


class Trainer(object):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            model_class,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None,
            inner_dim=None,
            RoPE=False,

    ):

        self.args = args
        self.model = model
        self.data_processor = data_processor
        self.tokenizer = tokenizer

        if train_dataset is not None and isinstance(train_dataset, Dataset):
            self.train_dataset = train_dataset

        if eval_dataset is not None and isinstance(eval_dataset, Dataset):
            self.eval_dataset = eval_dataset

        self.logger = logger
        self.model_class = model_class
        self.ngram_dict = ngram_dict
        self.inner_dim = inner_dim
        self.RoPE = RoPE

    def train(self, train_dataset=None, eval_dataset=None):
        args = self.args
        logger = self.logger
        model = self.model
        model.to(args.device)
        if train_dataset:
            self.train_dataset = train_dataset
            self.eval_dataset = eval_dataset
        train_dataloader = self.get_train_dataloader()

        num_training_steps = len(train_dataloader) * args.epochs
        num_warmup_steps = num_training_steps * args.warmup_proportion
        num_examples = len(train_dataloader.dataset)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps)

        logger.info('***** Running training *****')
        logger.info('Num samples %d', num_examples)
        logger.info('Num epochs %d', args.epochs)
        logger.info('Num training steps %d', num_training_steps)
        logger.info('Num warmup steps %d', num_warmup_steps)

        global_step = 0
        best_step = None
        best_p = .0
        best_r = .0
        best_score = .0
        cnt_patience = 0
        writer = SummaryWriter(args.output_dir + '/logs')
        for i in range(args.epochs):
            pbar = ProgressBar(n_total=len(train_dataloader), desc="Training")
            for step, item in enumerate(train_dataloader):
                loss = self.training_step(model, item)
                pbar(step, {'loss': loss.item()})
                writer.add_scalar('Loss', loss, i*len(train_dataloader)+step)

                if args.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    print('')
                    p, r, f1, preds_num = self.evaluate(model)
                    if f1 > best_score:
                        best_p = p
                        best_r = r
                        best_score = f1
                        best_step = global_step
                        cnt_patience = 0
                        self._save_checkpoint(model, global_step)
                        self._save_real_predict(preds_num, 'eval')
                    else:
                        cnt_patience += 1
                        self.logger.info('Earlystopper counter: %s out of %s', cnt_patience, args.earlystop_patience)
                        if cnt_patience >= self.args.earlystop_patience:
                            break
            if cnt_patience >= args.earlystop_patience:
                break
        if best_step:
            logger.info('Training Stop! The best step %s: %s', best_step, best_score)
            if args.device == 'cuda':
                torch.cuda.empty_cache()
            self._save_best_checkpoint(best_step=best_step)
        else:
            self._save_checkpoint(model, global_step)
            best_p, best_r, best_score, preds_num = self.evaluate(model)
            logger.info('Training Stop! The best step %s: %s', global_step, best_score)
            if args.device == 'cuda':
                torch.cuda.empty_cache()
            self._save_real_predict(preds_num, 'eval')
            self._save_best_checkpoint(best_step=global_step)
        writer.close()
        return best_p, best_r, best_score

    def evaluate(self, model):
        raise NotImplementedError

    def predict(self, model, test_dataset):
        raise NotImplementedError

    def _save_checkpoint(self, model, step):
        raise NotImplementedError

    def _save_best_checkpoint(self, best_step):
        raise NotImplementedError

    def _save_real_predict(self, preds_num, mode):
        samples = self.data_processor.get_dev_sample() if mode == 'eval' else self.data_processor.get_test_sample()
        output_name = 'eval_real_predict.tsv' if mode == 'eval' else 'test_real_predict.tsv'
        pred = ''
        entity = [samples['text'][i] for i in range(len(samples['text']))]
        label = [samples['label'][i] for i in range(len(samples['label']))]
        length = [len(sample) for sample in entity]
        length = sum(length)
        assert len(preds_num) == length
        pre = 0
        for i in range(len(entity)):
            for j in range(len(entity[i])):
                if self.data_processor.predict:
                    pred += entity[i][j] + '\t' + self.data_processor.id2label[preds_num[pre]] + '\n'
                else:
                    pred += entity[i][j] + '\t' + label[i][j] + '\t' + \
                            self.data_processor.id2label[preds_num[pre]] + '\n'
                pre += 1
            pred += '\n'
        pred = pred.strip()
        output_dir = os.path.join(self.args.output_dir, output_name)
        with open(output_dir, 'w', encoding='utf-8') as f:
            f.write(pred)
        return

    def training_step(self, model, item):
        raise NotImplementedError

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True
        )

    def get_eval_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False
        )

    def get_test_dataloader(self, test_dataset, batch_size=None):
        if not batch_size:
            batch_size = self.args.eval_batch_size

        return DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )


class BioSEP_Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            model_class,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None,
    ):
        super(BioSEP_Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
            model_class=model_class,
            ngram_dict=ngram_dict,
        )

        self.loss_bce = nn.BCELoss()
        self.loss_cross = nn.CrossEntropyLoss()

    def training_step(self, model, item):
        model.train()

        input_ids = item[0].to(self.args.device, dtype=torch.long)
        token_type_ids = item[1].to(self.args.device, dtype=torch.long)
        attention_mask = item[2].to(self.args.device, dtype=torch.long)
        # [-100, 0, 0, 0, 1, 2, 2, 2, 2, 2, -100]
        labels = item[3].to(self.args.device, dtype=torch.long)
        # [-100, 0, -100, -100, 1, -100, -100, 2, -100, -100, -100]
        label_mask_ids = item[4].to(self.args.device, dtype=torch.long)
        B_start_ids = item[5].to(self.args.device, dtype=torch.long)
        B_end_ids = item[6].to(self.args.device, dtype=torch.long)
        I_start_ids = item[7].to(self.args.device, dtype=torch.long)
        I_end_ids = item[8].to(self.args.device, dtype=torch.long)

        outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                        labels=(labels, B_start_ids, B_end_ids, I_start_ids, I_end_ids), label_mask=labels)
        loss = outputs[0]
        loss.backward()
        return loss.detach()

    def evaluate(self, model):
        args = self.args
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)

        preds_bert = None
        preds_SEP = None
        eval_labels_mask = None
        logger.info('***** Running evaluation *****')
        logger.info('Num samples %d', num_examples)
        for step, item in enumerate(eval_dataloader):
            model.eval()

            input_ids = item[0].to(self.args.device, dtype=torch.long)
            token_type_ids = item[1].to(self.args.device, dtype=torch.long)
            attention_mask = item[2].to(self.args.device, dtype=torch.long)
            labels = item[3].to(self.args.device, dtype=torch.long)
            label_mask = item[4].to(self.args.device, dtype=torch.long)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, token_type_ids=token_type_ids,
                                attention_mask=attention_mask)
                logits = outputs[1]
                B_start_logits, B_end_logits, I_start_logits, I_end_logits = outputs[2:6]
                pred_SEP = self.SEtrans(B_start_logits, B_end_logits, I_start_logits, I_end_logits,
                                        self.args.max_length)
                active_mask = label_mask != -100
                active_mask_logits = logits[active_mask]
                active_mask_logits_SEP = torch.tensor(pred_SEP).to(self.args.device)[active_mask.view(-1)]
                active_mask_labels = labels[active_mask]

            if preds_bert is None:
                preds_bert = active_mask_logits.detach().cpu().numpy()
                preds_SEP = active_mask_logits_SEP.detach().cpu().numpy()
                eval_labels_mask = active_mask_labels.detach().cpu().numpy()
            else:
                preds_bert = np.append(preds_bert, active_mask_logits.detach().cpu().numpy(), axis=0)
                preds_SEP = np.append(preds_SEP, active_mask_logits_SEP.detach().cpu().numpy(), axis=0)
                eval_labels_mask = np.append(eval_labels_mask, active_mask_labels.detach().cpu().numpy(), axis=0)

        preds_bert = np.argmax(preds_bert, axis=-1)
        p_mask, r_mask, f1_mask, _ = precision_recall_fscore_support(y_pred=preds_bert, y_true=eval_labels_mask,
                                                                     average='macro')
        logger.info('%s-%s true precision: %s - true recall: %s - true f1 score: %s', args.task_name, args.model_name,
                    p_mask, r_mask, f1_mask)
        # strict match
        conll_data = ['data ' + self.data_processor.id2label[eval_labels_mask[i]] + ' ' + \
                      self.data_processor.id2label[preds_SEP[i]] for i in range(len(preds_SEP))]
        (p_strict, r_strict, f1_strict) = conll_eval(conll_data).evaluate_conll_file()
        return p_strict, r_strict, f1_strict, preds_SEP

    def predict(self, model, test_dataset):
        args = self.args
        logger = self.logger
        test_dataloader = self.get_test_dataloader(test_dataset)
        num_examples = len(test_dataloader.dataset)
        model.to(args.device)

        preds_bert = None
        preds_SEP = None
        p_mask, r_mask, f1_mask = .0, .0, .0
        p_strict, r_strict, f1_strict = .0, .0, .0
        logger.info('***** Running prediction *****')
        logger.info('Num samples %d', num_examples)
        pbar = ProgressBar(n_total=len(test_dataloader), desc='Prediction')
        for step, item in enumerate(test_dataloader):
            model.eval()
            input_ids = item[0].to(self.args.device, dtype=torch.long)
            token_type_ids = item[1].to(self.args.device, dtype=torch.long)
            attention_mask = item[2].to(self.args.device, dtype=torch.long)
            labels = item[3].to(self.args.device, dtype=torch.long)
            label_mask = item[4].to(self.args.device, dtype=torch.long)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, token_type_ids=token_type_ids,
                                attention_mask=attention_mask)
                logits = outputs[1]
                B_start_logits, B_end_logits, I_start_logits, I_end_logits = outputs[2:6]
                pred_SEP = self.SEtrans(B_start_logits, B_end_logits, I_start_logits, I_end_logits,
                                        self.args.max_length)
                # pred_SEP: [batch_size * seq_length]
                active_mask = label_mask != -100
                active_mask_logits = logits[active_mask]
                active_mask_logits_SEP = torch.tensor(pred_SEP).to(self.args.device)[active_mask.view(-1)]
                active_mask_labels = labels[active_mask]

                if self.data_processor.predict:
                    if preds_bert is None:
                        preds_bert = active_mask_logits.detach().cpu().numpy()
                        preds_SEP = active_mask_logits_SEP.detach().cpu().numpy()
                    else:
                        preds_bert = np.append(preds_bert, active_mask_logits.detach().cpu().numpy(), axis=0)
                        preds_SEP = np.append(preds_SEP, active_mask_logits_SEP.detach().cpu().numpy(), axis=0)
                else:
                    # labels = item[3].to(self.args.device, dtype=torch.long)
                    if preds_bert is None:
                        preds_bert = active_mask_logits.detach().cpu().numpy()
                        preds_SEP = active_mask_logits_SEP.detach().cpu().numpy()
                        test_labels_mask = active_mask_labels.detach().cpu().numpy()
                    else:
                        preds_bert = np.append(preds_bert, active_mask_logits.detach().cpu().numpy(), axis=0)
                        preds_SEP = np.append(preds_SEP, active_mask_logits_SEP.detach().cpu().numpy(), axis=0)
                        test_labels_mask = np.append(test_labels_mask, active_mask_labels.detach().cpu().numpy(),
                                                     axis=0)
            pbar(step=step, info='')

        preds_bert = torch.sigmoid(torch.tensor(preds_bert)).numpy()
        preds_bert = preds_bert / preds_bert.sum(axis=1, keepdims=1)
        preds_num_mask = np.argmax(preds_bert, axis=-1)
        preds_pro_mask = [preds_bert[i][preds_num_mask[i]] for i in range(len(preds_num_mask))]
        #'''
        self._save_real_predict(preds_num_mask, 'test')
        if not self.data_processor.predict:
            preds_bert = np.argmax(preds_bert, axis=-1)
            p_mask, r_mask, f1_mask, _ = precision_recall_fscore_support(y_pred=preds_bert, y_true=test_labels_mask,
                                                                         average='macro')
            # strict match
            conll_data = ['data ' + self.data_processor.id2label[test_labels_mask[i]] + ' ' + \
                          self.data_processor.id2label[preds_SEP[i]] for i in range(len(preds_SEP))]
            (p_strict, r_strict, f1_strict) = conll_eval(conll_data).evaluate_conll_file()
        #'''
        return p_strict, r_strict, f1_strict, preds_SEP,

    def SEtrans(self, B_start_logits,B_end_logits,  I_start_logits, I_end_logits, max_seq):
        b_start_ids = [1 if i >= 0.5 else 0 for i in B_start_logits.view(-1)]
        b_end_ids = [1 if i >= 0.5 else 0 for i in B_end_logits.view(-1)]
        i_start_ids = [1 if i >= 0.5 else 0 for i in I_start_logits.view(-1)]
        i_end_ids = [1 if i >= 0.5 else 0 for i in I_end_logits.view(-1)]
        b_start_end_tuple_list = self.extract_entity(b_start_ids, b_end_ids, max_seq)
        i_start_end_tuple_list = self.extract_entity(i_start_ids, i_end_ids, max_seq)
        preds = [0] * len(B_start_logits.view(-1))
        for i, j in i_start_end_tuple_list:
            preds[i] = 2
            for k in range(i+1, j+1):
                preds[k] = 2
        for i, j in b_start_end_tuple_list:
            preds[i] = 1
            for k in range(i+1, j+1):
                preds[k] = 2
        return preds

    def extract_entity(self, start_ids, end_ids, max_seq):
        start_end_tuple_list = []
        for (i, start_id) in enumerate(start_ids):
            if start_id == 0:
                continue
            if end_ids[i] == 1:
                start_end_tuple_list.append((i, i))
                continue
            j = i + 1
            find_end_tag = False
            while j < len(end_ids) and j % max_seq:
                if start_ids[j] == 1:
                    break
                if end_ids[j] == 1:
                    start_end_tuple_list.append((i, j))
                    find_end_tag = True
                    break
                else:
                    j += 1
            if not find_end_tag:
                start_end_tuple_list.append((i, i))
        return start_end_tuple_list

    def _save_checkpoint(self, model, step):
        output_dir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.pth'))
        self.tokenizer.save_vocabulary(output_dir)
        self.logger.info('Saving models checkpoint to %s', output_dir)

    def _save_best_checkpoint(self, best_step):
        self.model.load_state_dict(torch.load(os.path.join(self.args.output_dir, f'checkpoint-{best_step}',
                                                           'pytorch_model.pth')))

        torch.save(self.model.state_dict(), os.path.join(self.args.output_dir, 'pytorch_model.pth'))
        self.tokenizer.save_vocabulary(self.args.output_dir)
        self.logger.info('Saving models checkpoint to %s', self.args.output_dir)
