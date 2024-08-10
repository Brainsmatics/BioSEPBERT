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
from torch.utils.tensorboard import SummaryWriter
from NER.utils import ProgressBar
from NER.conlleval import conll_eval
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score, recall_score, f1_score


def re_metric(preds, labels):
    return precision_recall_fscore_support(y_pred=preds, y_true=labels, average='macro')


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
        preds_num = None
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
                writer.add_scalar('loss', loss, i)

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
        return best_p, best_r, best_score, preds_num

    def evaluate(self, model):
        raise NotImplementedError

    def predict(self, model, test_dataset):
        raise NotImplementedError

    def _save_checkpoint(self, model, step):
        raise NotImplementedError

    def _save_best_checkpoint(self, best_step):
        raise NotImplementedError

    def _save_real_predict(self, preds_num, mode):
        raise NotImplementedError

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


class RETrainer(Trainer):
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
            ngram_dict=None
    ):
        super(RETrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
            model_class=model_class,
            ngram_dict=ngram_dict
        )

    def training_step(self, model, item):
        model.train()

        input_ids, token_type_ids, attention_mask, flag, label = item

        input_ids, token_type_ids, attention_mask, flag, label = input_ids.to(self.args.device), \
                                                                 token_type_ids.to(self.args.device), \
                                                                 attention_mask.to(self.args.device), \
                                                                 flag.to(self.args.device), label.to(self.args.device)

        loss, logits = model(input_ids, token_type_ids, attention_mask, flag, label)
        loss.backward()

        return loss.detach()

    def evaluate(self, model):
        args = self.args
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)

        preds = None
        eval_labels = None

        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc="Eval")
        for step, item in enumerate(eval_dataloader):
            model.eval()

            input_ids, token_type_ids, attention_mask, flag, label = item

            input_ids, token_type_ids, attention_mask, flag, label = input_ids.to(self.args.device), \
                                                                     token_type_ids.to(self.args.device), \
                                                                     attention_mask.to(self.args.device), \
                                                                     flag.to(self.args.device), label.to(self.args.device)

            with torch.no_grad():
                loss, logits = model(input_ids, token_type_ids, attention_mask, flag, label)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                eval_labels = label.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                eval_labels = np.append(eval_labels, label.detach().cpu().numpy(), axis=0)
            pbar(step=step, info='')

        preds = np.argmax(preds, axis=1)
        p = precision_score(eval_labels, preds, average='macro')
        r = recall_score(eval_labels, preds, average='macro')
        f1 = f1_score(eval_labels, preds, average='macro')
        # p, r, f1, _ = re_metric(preds, eval_labels)
        logger.info("%s-%s precision: %s - recall: %s - f1 score: %s", args.task_name, args.model_name, p, r, f1)
        return p, r, f1, preds

    def predict(self, model, test_samples):
        args = self.args
        logger = self.logger
        test_dataloader = self.get_test_dataloader(test_samples)
        num_examples = len(test_dataloader.dataset)
        model.to(args.device)

        preds = None
        test_labels = None

        logger.info("***** Running prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(test_dataloader), desc="Prediction")
        for step, item in enumerate(test_dataloader):
            model.eval()

            input_ids, token_type_ids, attention_mask, flag, label = item

            input_ids, token_type_ids, attention_mask, flag, label = input_ids.to(self.args.device), \
                                                                     token_type_ids.to(self.args.device), \
                                                                     attention_mask.to(self.args.device), \
                                                                     flag.to(self.args.device), label.to(self.args.device)

            with torch.no_grad():
                loss, logits = model(input_ids, token_type_ids, attention_mask, flag, label)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                test_labels = label.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                test_labels = np.append(test_labels, label.detach().cpu().numpy(), axis=0)
            pbar(step=step, info='')

        preds = np.argmax(preds, axis=1)
        p = precision_score(test_labels, preds, average='macro')
        r = recall_score(test_labels, preds, average='macro')
        f1 = f1_score(test_labels, preds, average='macro')
        # p, r, f1, _ = re_metric(preds, test_labels)
        self._save_real_predict(preds, 'test')
        logger.info("%s-%s precision: %s - recall: %s - f1 score: %s", args.task_name, args.model_name, p, r, f1)
        return p, r, f1, preds

    def _save_checkpoint(self, model, step):
        output_dir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.pth'))
        self.tokenizer.save_vocabulary(output_dir)
        self.logger.info('Saving models checkpoint to %s', output_dir)
        return

    def _save_best_checkpoint(self, best_step):
        self.model.load_state_dict(torch.load(os.path.join(self.args.output_dir, f'checkpoint-{best_step}',
                                                           'pytorch_model.pth')))

        torch.save(self.model.state_dict(), os.path.join(self.args.output_dir, 'pytorch_model.pth'))
        self.tokenizer.save_vocabulary(self.args.output_dir)
        self.logger.info('Saving models checkpoint to %s', self.args.output_dir)
        return

    def _save_real_predict(self, preds_num, mode):
        samples = self.data_processor.get_dev_sample() if mode == 'eval' else self.data_processor.get_test_sample()
        output_name = 'eval_real_predict.tsv' if mode == 'eval' else 'test_real_predict.tsv'
        pred = ''
        text = [samples['text'][i] for i in range(len(samples['text']))]
        label = [samples['label'][i] for i in range(len(samples['label']))]
        pre = 0
        for i in range(len(text)):
            if self.data_processor.predict:
                pred += text[i] + '\t' + str(preds_num[i]) + '\n'
            else:
                pred += text[i] + '\t' + str(label[i]) + '\t' + str(preds_num[i]) + '\n'
            pre += 1
        pred += '\n'
        pred = pred.strip()
        output_dir = os.path.join(self.args.output_dir, output_name)
        with open(output_dir, 'w', encoding='utf-8') as f:
            f.write(pred)
        return


class REDirectionTrainer(Trainer):
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
            ngram_dict=None
    ):
        super(REDirectionTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
            model_class=model_class,
            ngram_dict=ngram_dict
        )

    def training_step(self, model, item):
        model.train()

        input_ids, token_type_ids, attention_mask, flag, label = item

        input_ids, token_type_ids, attention_mask, flag, label = input_ids.to(self.args.device), \
                                                                 token_type_ids.to(self.args.device), \
                                                                 attention_mask.to(self.args.device), \
                                                                 flag.to(self.args.device), label.to(self.args.device)

        loss, logits = model(input_ids, token_type_ids, attention_mask, flag, label)
        loss.backward()

        return loss.detach()

    def evaluate(self, model):
        args = self.args
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)

        preds = None
        eval_labels = None

        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc="Eval")
        for step, item in enumerate(eval_dataloader):
            model.eval()

            input_ids, token_type_ids, attention_mask, flag, label = item

            input_ids, token_type_ids, attention_mask, flag, label = input_ids.to(self.args.device), \
                                                                     token_type_ids.to(self.args.device), \
                                                                     attention_mask.to(self.args.device), \
                                                                     flag.to(self.args.device), label.to(self.args.device)

            with torch.no_grad():
                loss, logits = model(input_ids, token_type_ids, attention_mask, flag, label)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                eval_labels = label.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                eval_labels = np.append(eval_labels, label.detach().cpu().numpy(), axis=0)
            pbar(step=step, info='')

        preds = np.argmax(preds, axis=1)
        p = precision_score(eval_labels, preds, average='macro')
        r = recall_score(eval_labels, preds, average='macro')
        f1 = f1_score(eval_labels, preds, average='macro')
        # p, r, f1, _ = re_metric(preds, eval_labels)
        logger.info("%s-%s precision: %s - recall: %s - f1 score: %s", args.task_name, args.model_name, p, r, f1)
        return p, r, f1, preds

    def predict(self, model, test_samples):
        args = self.args
        logger = self.logger
        test_dataloader = self.get_test_dataloader(test_samples)
        num_examples = len(test_dataloader.dataset)
        model.to(args.device)

        preds = None
        test_labels = None

        logger.info("***** Running prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(test_dataloader), desc="Prediction")
        for step, item in enumerate(test_dataloader):
            model.eval()

            input_ids, token_type_ids, attention_mask, flag, label = item

            input_ids, token_type_ids, attention_mask, flag, label = input_ids.to(self.args.device), \
                                                                     token_type_ids.to(self.args.device), \
                                                                     attention_mask.to(self.args.device), \
                                                                     flag.to(self.args.device), label.to(self.args.device)

            with torch.no_grad():
                loss, logits = model(input_ids, token_type_ids, attention_mask, flag, label)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                test_labels = label.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                test_labels = np.append(test_labels, label.detach().cpu().numpy(), axis=0)
            pbar(step=step, info='')

        preds = np.argmax(preds, axis=1)
        p = precision_score(test_labels, preds, average='macro')
        r = recall_score(test_labels, preds, average='macro')
        f1 = f1_score(test_labels, preds, average='macro')
        # p, r, f1, _ = re_metric(preds, test_labels)
        #self._save_real_predict(preds, 'test')
        logger.info("%s-%s precision: %s - recall: %s - f1 score: %s", args.task_name, args.model_name, p, r, f1)
        return p, r, f1, preds

    def _save_checkpoint(self, model, step):
        output_dir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.pth'))
        self.tokenizer.save_vocabulary(output_dir)
        self.logger.info('Saving models checkpoint to %s', output_dir)
        return

    def _save_best_checkpoint(self, best_step):
        self.model.load_state_dict(torch.load(os.path.join(self.args.output_dir, f'checkpoint-{best_step}',
                                                           'pytorch_model.pth')))

        torch.save(self.model.state_dict(), os.path.join(self.args.output_dir, 'pytorch_model.pth'))
        self.tokenizer.save_vocabulary(self.args.output_dir)
        self.logger.info('Saving models checkpoint to %s', self.args.output_dir)
        return

    def _save_real_predict(self, preds_num, mode):
        samples = self.data_processor.get_dev_sample() if mode == 'eval' else self.data_processor.get_test_sample()
        output_name = 'eval_real_predict.tsv' if mode == 'eval' else 'test_real_predict.tsv'
        pred = ''
        text = [samples['text'][i] for i in range(len(samples['text']))]
        label = [samples['label'][i] for i in range(len(samples['label']))]
        pre = 0
        for i in range(len(text)):
            if self.data_processor.predict:
                pred += text[i] + '\t' + str(preds_num[i]) + '\n'
            else:
                pred += text[i] + '\t' + str(label[i]) + '\t' + str(preds_num[i]) + '\n'
            pre += 1
        pred += '\n'
        pred = pred.strip()
        output_dir = os.path.join(self.args.output_dir, output_name)
        with open(output_dir, 'w', encoding='utf-8') as f:
            f.write(pred)
        return

