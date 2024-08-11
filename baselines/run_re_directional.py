import os
import sys
sys.path.append('.')
import copy
import argparse
import xlwt
import torch
from transformers import BertTokenizer, BertModel

from NRE.model.modeling import REDirectionModel
from NRE.trainer.trainer import REDirectionTrainer
from NRE.utils import init_logger, seed_everything
from NRE.data.dataset import REDataset
from NRE.data.data_process import RESLAUGDirectionDataProcessor


MODEL_CLASS = {
    'BioSEPBERT': (BertTokenizer, BertModel),
    'biobert': (BertTokenizer, BertModel),
    'PubMedBERT': (BertTokenizer, BertModel),
    'scibert': (BertTokenizer, BertModel),
    'ClinicalBERT': (BertTokenizer, BertModel),
}


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default=None, type=str, required=True,
                    help="The task data directory.")
parser.add_argument("--model_dir", default=None, type=str, required=True,
                    help="The directory of pretrained models")
parser.add_argument("--model_type", default=None, type=str, required=True,
                    help="The type of selected pretrained models.")
parser.add_argument("--model_name", default=None, type=str, required=True,
                    help="The path of selected pretrained models. (e.g. chinese-bert-wwm)")
parser.add_argument("--task_name", default=None, type=str, required=True,
                    help="The name of task to train")
parser.add_argument("--output_dir", default=None, type=str, required=True,
                    help="The path of result data and models to be saved.")
parser.add_argument("--do_train", action='store_true',
                    help="Whether to run training.")
parser.add_argument("--do_predict", action='store_true',
                    help="Whether to run the models in inference mode on the test set.")
parser.add_argument("--augmentation", action='store_true',
                    help="Whether to add and delete the dictionary on the set.")
parser.add_argument("--repre", action='store_true',
                    help="Whether to add different entity pairs on the set.")
parser.add_argument("--result_output_dir", default=None, type=str,
                    help="the directory of commit result to be saved")

# models param
parser.add_argument("--max_length", default=128, type=int,
                    help="the max length of sentence.")
parser.add_argument("--train_batch_size", default=8, type=int,
                    help="Batch size for training.")
parser.add_argument("--eval_batch_size", default=8, type=int,
                    help="Batch size for evaluation.")
parser.add_argument("--learning_rate", default=5e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.01, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--epochs", default=3, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion", default=0.1, type=float,
                    help="Proportion of training to perform linear learning rate warmup for, "
                         "E.g., 0.1 = 10% of training.")
parser.add_argument("--earlystop_patience", default=2, type=int,
                    help="The patience of early stop")

parser.add_argument('--logging_steps', type=int, default=10,
                    help="Log every X updates steps.")
parser.add_argument('--save_steps', type=int, default=1000,
                    help="Save checkpoint every X updates steps.")
parser.add_argument('--seed', type=int, default=2021,
                    help="random seed for initialization")

args = parser.parse_args()


def main(num):
    args.output_dir = os.path.join(args.output_dir, 're')
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    seed_everything(args.seed)

    tokenizer_class, model_class = MODEL_CLASS[args.model_type]
    p_re_eval, r_re_eval, f_re_eval = 0, 0, 0
    p_re_test, r_re_test, f_re_test = 0, 0, 0

    if args.do_train:
        args.output_dir = os.path.join(output_dir, 're', str(num))
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        logger = init_logger(os.path.join(args.output_dir, f'{args.task_name}_{args.model_name}.log'))
        logger.info('Training RE model...')
        logger.info('Training/evaluation parameters %s', args)
        tokenizer = tokenizer_class.from_pretrained(os.path.join(args.model_dir, args.model_name))
        tokenizer.add_special_tokens({'additional_special_tokens': ['<s>', '</s>', '<o>', '</o>']})

        data_processor = RESLAUGDirectionDataProcessor(root=args.data_dir, repre=args.repre)
        train_samples = data_processor.get_train_sample()
        eval_samples = data_processor.get_dev_sample()
        train_dataset = REDataset(train_samples, data_processor, tokenizer, mode='train', model_type=args.model_type,
                                  max_length=args.max_length)
        eval_dataset = REDataset(eval_samples, data_processor, tokenizer, mode='eval', model_type=args.model_type,
                                 max_length=args.max_length)
        model = model_class.from_pretrained(os.path.join(args.model_dir, args.model_name),
                                            num_labels=data_processor.num_labels)
        model = REDirectionModel(tokenizer, model, num_labels=data_processor.num_labels)
        trainer = REDirectionTrainer(args=args, model=model, data_processor=data_processor,
                                   tokenizer=tokenizer, train_dataset=train_dataset, eval_dataset=eval_dataset,
                                   logger=logger, model_class=REDirectionModel)

        p_re_eval, r_re_eval, f_re_eval, _ = trainer.train()

    if args.do_predict:
        args.output_dir = os.path.join(output_dir, 're', str(num))
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        logger = init_logger(os.path.join(args.output_dir, f'{args.task_name}_{args.model_name}.log'))
        logger.info('Training/evaluation parameters %s', args)
        tokenizer = tokenizer_class.from_pretrained(os.path.join(args.model_dir, args.model_name))
        tokenizer.add_special_tokens({'additional_special_tokens': ['<s>', '</s>', '<o>', '</o>']})
        data_processor = RESLAUGDirectionDataProcessor(root=args.data_dir, repre=args.repre)
        test_samples = data_processor.get_test_sample()
        test_dataset = REDataset(test_samples, data_processor, tokenizer, mode='test', model_type=args.model_type,
                                 max_length=args.max_length)

        model = model_class.from_pretrained(os.path.join(args.model_dir, args.model_name),
                                            num_labels=data_processor.num_labels)
        model = REDirectionModel(tokenizer, model, num_labels=data_processor.num_labels)
        model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name, 'pytorch_model.pth')))
        # If you want to train the model, please use the following line to do so
        # model.load_state_dict(torch.load(os.path.join(args.output_dir, 'pytorch_model.pth')))
        trainer = REDirectionTrainer(args=args, model=model, data_processor=data_processor,
                                   tokenizer=tokenizer, logger=logger, model_class=REDirectionModel)
        p_re_test, r_re_test, f_re_test, _ = trainer.predict(model=model, test_samples=test_dataset)

    out_re = (p_re_eval, r_re_eval, f_re_eval, p_re_test, r_re_test, f_re_test)

    return out_re


if __name__ == '__main__':
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, args.task_name)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, args.model_name)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    output_dir = copy.copy(args.output_dir)

    workbook = xlwt.Workbook()
    sheet_re = workbook.add_sheet('RE', cell_overwrite_ok=True)
    for i in range(1, 11):
        args.data_dir = args.data_dir[:-1] + str(i)
        out_re = main(i)
        for j in range(6):
            sheet_re.write(i, j+1, out_re[j])
        output_file = os.path.join(output_dir, 'result.xls')
        workbook.save(output_file)
