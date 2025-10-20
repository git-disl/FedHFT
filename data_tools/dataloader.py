from __future__ import annotations

import os

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from data_tools.sampling import *

skip_exec = True


def prepare_datasets(model_name: str, task_name: str, data_name: str, tokenizer, cache_dir: str, eval_key: str = 'val'):
    if task_name == 'glue':
        if data_name == 'ag_news':
            return prepare_datasets_agnews(model_name, data_name, tokenizer, cache_dir, eval_key)
        else:
            return prepare_datasets_glue(model_name, data_name, tokenizer, cache_dir, eval_key)
    else:
        raise NotImplementedError


def prepare_datasets_agnews(model_name: str, data_name: str, tokenizer, cache_dir: str, eval_key: str = 'val'):
    sentence1_key, sentence2_key = ('text', None)

    # used to preprocess the raw data
    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (
                examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=False, max_length=128, truncation=True)

        if 'label' in examples:
            result['labels'] = examples['label']
        return result

    raw_datasets = load_dataset(data_name)

    column_names = raw_datasets['train'].column_names
    processed_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=column_names)

    validation_datasets = processed_datasets['test']

    return processed_datasets['train'], validation_datasets, validation_datasets


def prepare_datasets_glue(model_name: str, data_name: str, tokenizer, cache_dir: str, eval_key: str = 'val'):
    task_to_keys = {
        'cola': ('sentence', None),
        'mnli': ('premise', 'hypothesis'),
        'mrpc': ('sentence1', 'sentence2'),
        'qnli': ('question', 'sentence'),
        'qqp': ('question1', 'question2'),
        'rte': ('sentence1', 'sentence2'),
        'sst2': ('sentence', None),
        'stsb': ('sentence1', 'sentence2'),
        'wnli': ('sentence1', 'sentence2'),
        'ag_news': ('text', None)
    }

    sentence1_key, sentence2_key = task_to_keys[data_name]

    # used to preprocess the raw data
    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (
            examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, max_length=128, padding=True, truncation=True)

        if 'label' in examples:
            result['labels'] = examples['label']
        return result

    if data_name == 'ag_news':
        raw_datasets = load_dataset(data_name)
    else:
        raw_datasets = load_dataset('glue', data_name)

    if eval_key == 'val':
        for key in list(raw_datasets.keys()):
            if 'test' in key:
                raw_datasets.pop(key)

    column_names = raw_datasets['train'].column_names
    processed_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=column_names)

    if data_name == 'mnli':
        if eval_key == 'test':
            validation_datasets = processed_datasets['test_matched']
        else:
            validation_datasets = processed_datasets['validation_matched']
    else:
        if eval_key == 'test':
            validation_datasets = processed_datasets['test']

        else:
            validation_datasets = processed_datasets['validation']

    return processed_datasets['train'], validation_datasets, validation_datasets


def get_user_groups(data_content, args):
    train_user_groups, val_user_groups, test_user_groups, stats_df = create_noniid_users(data_content, args, args.alpha)
    return train_user_groups, val_user_groups, test_user_groups, stats_df


def get_dataloaders(args, batch_size, dataset):
    train_loader, val_loader, test_loader = None, None, None
    train_set, val_set, test_set = dataset

    if args.use_valid:
        if val_set is None:
            train_set_index = torch.randperm(len(train_set))
            if os.path.exists(os.path.join(args.save_path, 'index.pth')):
                train_set_index = torch.load(os.path.join(args.save_path, 'index.pth'))
            else:
                torch.save(train_set_index, os.path.join(args.save_path, 'index.pth'))

            if args.data == 'sst2':
                num_sample_valid = 872
            elif args.data == 'ag_news':
                num_sample_valid = 0
            else:
                raise NotImplementedError

            train_indices = train_set_index[:-num_sample_valid]
            val_indices = train_set_index[-num_sample_valid:]
            val_set = train_set
        else:
            train_indices = torch.arange(len(train_set))
            val_indices = torch.arange(len(val_set))

        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_indices),
                num_workers=args.workers, pin_memory=True)
        if 'val' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    val_indices),
                num_workers=args.workers, pin_memory=True)
        if 'test' in args.splits:
            test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
    else:
        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
        if 'val' or 'test' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
            test_loader = val_loader

    if 'train' not in args.splits:
        if len(val_loader.dataset.transform.transforms) > 2:
            val_loader.dataset.transform.transforms = val_loader.dataset.transform.transforms[-2:]

    if 'bert' in args.arch:
        train_loader.collate_fn = collate_fn
        val_loader.collate_fn = collate_fn
        test_loader.collate_fn = collate_fn

    return train_loader, val_loader, test_loader


def get_client_dataloader(dataset, idxs, args, batch_size):
    """
    Returns train, validation and test dataloaders for a given dataset
    and user indexes.
    """
    if 'bert' in args.arch:
        return torch.utils.data.DataLoader(dataset, batch_size=min(batch_size, len(idxs)),
                                         sampler=torch.utils.data.sampler.SubsetRandomSampler(idxs),
                                         num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=min(batch_size, len(idxs)),
                                         sampler=torch.utils.data.sampler.SubsetRandomSampler(idxs),
                                         num_workers=args.workers, pin_memory=True)


def collate_fn(data):
    return (pad_sequence([torch.tensor(d['input_ids']) for d in data], batch_first=True, padding_value=0),
            torch.tensor([d['label'] for d in data]))
