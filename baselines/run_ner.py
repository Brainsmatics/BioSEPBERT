# encoding: utf-8
# @author: xkchai
# @license: (C) Copyright 2020-2024, Node Supply Chain Manager Corporation Limited.
# @contact: chaixiaokang@hust.edu.cn

import os
import sys
import argparse
import torch
import xlwt
import numpy as np
from transformers import BertTokenizer, BertModel
from NER.utils import init_logger, seed_everything
from NER.data.dataset import BioSEPDataset
from NER.data.data_process import NERDataProcessor
from NER.trainer.trainer import BioSEP_Trainer
from NER.model.modeling import BioSEPBERT_Model


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

NER_MODEL_CLASS = {
    'BioSEPBERT': (BertTokenizer, BertModel),
    'biobert': (BertTokenizer, BertModel),
    'PubMedBERT': (BertTokenizer, BertModel),
    'scibert': (BertTokenizer, BertModel),
    'ClinicalBERT': (BertTokenizer, BertModel),
}

NER_CUSTOM_MODEL_CLASS = {
    'BioSEPBERT': BioSEPBERT_Model,
}

TASK_DATASET_CLASS = {
    'BioSEPBERT': (BioSEPDataset, NERDataProcessor),
}

TASK_TRAINER = {
    'BioSEPBERT': BioSEP_Trainer,
}

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default=None, type=str, required=True,
                    help='The task data directory')
parser.add_argument('--model_dir', default=None, type=str, required=True,
                    help='The directory of pretrained models')
parser.add_argument('--model_type', default=None, type=str, required=True,
                    help='The type of selected pretrained models')
parser.add_argument("--model_name", default=None, type=str, required=True,
                    help="The path of selected pretrained models. (e.g. biobert)")
parser.add_argument("--task_name", default='BR_NER', type=str, required=True,
                    help="The name of task to train")
parser.add_argument("--output_dir", default=None, type=str, required=True,
                    help="The path of result data and models to be saved.")
parser.add_argument("--do_train", action='store_true',
                    help="Whether to run training.")
parser.add_argument("--do_predict", action='store_true',
                    help="Whether to run the models in inference mode on the test set.")
parser.add_argument("--result_output_dir", default=None, type=str,
                    help="the directory of commit result to be saved")
parser.add_argument("--cross_validation", action='store_true',
                    help="Whether use cross validation")

# models param
parser.add_argument("--max_length", default=128, type=int,
                    help="the max length of sentence.")
parser.add_argument("--train_batch_size", default=16, type=int,
                    help="Batch size for training.")
parser.add_argument("--eval_batch_size", default=16, type=int,
                    help="Batch size for evaluation.")
parser.add_argument("--learning_rate", default=3e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.01, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--epochs", default=5, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion", default=0.1, type=float,
                    help="Proportion of training to perform linear learning rate warmup for, "
                         "E.g., 0.1 = 10% of training.")
parser.add_argument("--earlystop_patience", default=200, type=int,
                    help="The patience of early stop")

parser.add_argument('--logging_steps', type=int, default=-1,
                    help="Log every X updates steps.")
parser.add_argument('--save_steps', type=int, default=1000,
                    help="Save checkpoint every X updates steps.")
parser.add_argument('--seed', type=int, default=2021,
                    help="random seed for initialization")
args = parser.parse_args()


def main(num):
    # add num folder
    args.output_dir = args.output_dir[:-1] + str(num)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    logger = init_logger(os.path.join(args.output_dir, f'{args.task_name}_{args.model_name}.log'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    seed_everything(args.seed)

    tokenizer_class, model_class = NER_MODEL_CLASS[args.model_type]
    dataset_class, data_processor_class = TASK_DATASET_CLASS[args.task_name]
    trainer_class = TASK_TRAINER[args.task_name]
    custom_model_class = NER_CUSTOM_MODEL_CLASS[args.task_name]

    logger.info('Training/evaluation parameters %s', args)

    data_processor = data_processor_class(root=args.data_dir, augmentation=False)
    tokenizer = tokenizer_class.from_pretrained(os.path.join(args.model_dir, args.model_name))
    model = model_class.from_pretrained(os.path.join(args.model_dir, args.model_name),
                                        num_labels=data_processor.num_labels)
    model = custom_model_class(model)
    trainer = trainer_class(args=args, model=model, data_processor=data_processor, tokenizer=tokenizer,
                            logger=logger, model_class=model_class)
    output = str(num) + '\n'
    eval_p, eval_r, eval_f1, test_p, test_r, test_f1 = 0, 0, 0, 0, 0, 0
    if args.do_train:
        train_samples = data_processor.get_train_sample()
        eval_samples = data_processor.get_dev_sample()
        train_dataset = dataset_class(train_samples, data_processor, tokenizer, mode='train',
                                      model_type=args.model_type, max_length=args.max_length)
        eval_dataset = dataset_class(eval_samples, data_processor, tokenizer, mode='eval',
                                     model_type=args.model_type, max_length=args.max_length)
        eval_p, eval_r, eval_f1 = trainer.train(train_dataset, eval_dataset)
        output += "Eval results: Precision: %6.2f%%; Recall: %6.2f%%; F1: %6.2f%%" % (eval_p, eval_r, eval_f1) + '\n'

    if args.do_predict:
        test_samples = data_processor.get_test_sample()
        test_dataset = dataset_class(test_samples, data_processor, tokenizer, mode='test',
                                     model_type=args.model_type, max_length=args.max_length)
        model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name, 'pytorch_model.pth')))
        # If you want to train the model, please use the following line to do so
        # model.load_state_dict(torch.load(os.path.join(args.output_dir, 'pytorch_model.pth')))
        test_p, test_r, test_f1, _ = trainer.predict(test_dataset=test_dataset, model=model)
        output += "Test results: Precision: %6.2f%%; Recall: %6.2f%%; F1: %6.2f%%" % (test_p, test_r, test_f1) + '\n'

    out_ner = (eval_p, eval_r, eval_f1, test_p, test_r, test_f1)

    return output, out_ner


if __name__ == '__main__':
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, args.task_name)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, args.model_name)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + '/0'

    workbook = xlwt.Workbook()
    sheet_ner = workbook.add_sheet('NER', cell_overwrite_ok=True)
    for i in range(1, 9):
        args.data_dir = args.data_dir[:-1] + str(i)
        _, out_ner = main(i)
        for j in range(6):
            sheet_ner.write(i, j + 1, out_ner[j])
        output_file = os.path.join(args.output_dir[:-1], 'result.xls')
        workbook.save(output_file)
