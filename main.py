#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim
from transformers import BertForSequenceClassification, BertForQuestionAnswering, \
    BertTokenizerFast, DataCollatorWithPadding

from args import arg_parser, modify_args
from config import Config
from data_tools.dataloader import prepare_datasets, get_user_groups
from fed import Federator
from paths import get_path
from utils.utils import load_checkpoint, save_user_groups, load_user_groups

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')
np.set_printoptions(precision=2)

args = arg_parser.parse_args()
args = modify_args(args)
torch.manual_seed(args.seed)

model_dispatcher = {
    'bert-base-uncased': {'glue': BertForSequenceClassification,
                          'qa': BertForQuestionAnswering},
    'bert-large-uncased': {'glue': BertForSequenceClassification,
                           'qa': BertForQuestionAnswering}
}

tokenizer_dispatcher = {
    'bert-base-uncased': BertTokenizerFast,
    'bert-large-uncased': BertTokenizerFast
}


def build_model(pretrained_model_name_or_path: str, task_name: str, data_name: str, **kwargs):
    is_regression = data_name == 'stsb'
    if is_regression:
        num_labels = 1
    else:
        if data_name == 'ag_news':
            num_labels = 4
        else:
            num_labels = 2

    if isinstance(model_dispatcher[pretrained_model_name_or_path], dict):
        model = model_dispatcher[pretrained_model_name_or_path][task_name].from_pretrained(
            pretrained_model_name_or_path,
            num_labels=num_labels, cache_dir='cache')
    else:
        model = model_dispatcher[pretrained_model_name_or_path].from_pretrained(pretrained_model_name_or_path,
                                                                                num_labels=num_labels,
                                                                                cache_dir='cache')
    return model


def prepare_data(args, eval_key):
    tokenizer_name = args.arch
    tokenizer = tokenizer_dispatcher[args.arch].from_pretrained(tokenizer_name, cache_dir='cache')
    train_dataset, validation_dataset, test_dataset = prepare_datasets(args.arch, args.task, args.data, tokenizer,
                                                                       args.data_root, eval_key)

    data_collator = DataCollatorWithPadding(tokenizer)

    return {'train': train_dataset, 'val': validation_dataset, 'test': test_dataset,
            'collator': data_collator, 'tokenizer': tokenizer}


def main():
    global args

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if os.path.exists(get_path(args, 'MAIN_FOLDER_DIR', temp=False)):
        shutil.rmtree(get_path(args, 'MAIN_FOLDER_DIR', temp=False))
    Path(get_path(args, 'TRAINER_FOLDER_DIR')).mkdir(exist_ok=True, parents=True)
    Path(get_path(args, 'MODEL_FOLDER_DIR')).mkdir(exist_ok=True, parents=True)
    Path(get_path(args, 'FIGURE_FOLDER_DIR')).mkdir(exist_ok=True, parents=True)

    config = Config(args)

    model = build_model(args.arch, args.task, args.data)

    if args.device == 'cuda':
        model = model.cuda()

    if args.resume:
        checkpoint = load_checkpoint(args, load_best=False)
        if checkpoint is not None:
            args.start_round = checkpoint['round'] + 1
            model.load_state_dict(checkpoint['state_dict'])

    cudnn.benchmark = True

    batch_size = args.batch_size if args.batch_size else config.get_init_training_params(args.arch, args.data)['batch_size']
    data_content = prepare_data(args, args.final_eval_split)

    train_user_groups, val_user_groups, test_user_groups, stats_df = get_user_groups(data_content, args)
    data_content['stats_df'] = stats_df

    prev_user_groups = load_user_groups(args)
    if prev_user_groups is None:
        if args.resume:
            print('Could not find user groups')
            raise RuntimeError
        user_groups = (train_user_groups, val_user_groups, test_user_groups)
        save_user_groups(args, (train_user_groups, val_user_groups, test_user_groups))
    else:
        user_groups = prev_user_groups

    with open(os.path.join(args.save_path, 'args.txt'), 'w') as f:
        print(args, file=f)

    federator = Federator(model, args)
    best_acc1, best_round = federator.fed_train(args, config, data_content, user_groups, batch_size,
                                                config.get_init_training_params(args.arch, args.data))

    print(f'best acc: {best_acc1}, best_round: {best_round}')


if __name__ == '__main__':
    main()
