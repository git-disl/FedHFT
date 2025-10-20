from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from transformers import EvalPrediction
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from datasets import load_metric
from paths import get_path


def prepare_traced_trainer(model, args, data_content, training_params={}, for_train_flag=True, for_eval_flag=True,
                           tag='default', device=None, eval_rate=1., send_tag='train', sample_idxs=[]):
    is_regression = args.data == 'stsb'

    if eval_rate == 1:
        val_sample_idxs = range(len(data_content['val']))
    else:
        val_sample_idxs = random.sample(range(len(data_content['val'])),
                                        int(len(data_content['val']) * eval_rate))

    try:
        round_idx = int(tag.split('_')[1])
    except:
        round_idx = 1

    save_strategy = 'no'
    evaluation_strategy = 'no'

    def compute_metrics(p: EvalPrediction):
        if args.task == 'glue':
            if args.data == 'ag_news':
                metric = load_metric('glue', 'sst2', trust_remote_code=True)
            else:
                metric = load_metric('glue', args.data, trust_remote_code=True)

            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=-1)
            result = metric.compute(predictions=preds, references=p.label_ids)

        else:
            raise NotImplementedError

        return result

    if tag == 'default':
        logging_dir = None
    else:
        logging_dir = get_path(args, 'TRAINER_FOLDER_DIR') + '/runs/' + tag

    if device is None:
        device = 'cuda'

    if device == 'cpu':
        no_cuda = True
    else:
        no_cuda = False

    if for_train_flag:
        for_eval_flag = False

    bf16 = False
    fp16 = False
    if 'q' in args.comp_method and 'finetune' in tag:
        if args.quant_method == 'bf16':
            bf16 = True
        if args.quant_method == 'fp16':
            fp16 = True

    learning_rate = training_params.get('learning_rate', 1e-4) * (0.9 ** round_idx)

    training_args = TrainingArguments(output_dir=get_path(args, 'TRAINER_FOLDER_DIR') + f'/runs/{tag}',
                                      do_train=for_train_flag,
                                      do_eval=for_eval_flag,
                                      evaluation_strategy=evaluation_strategy,
                                      save_strategy=save_strategy,
                                      logging_strategy='epoch',
                                      logging_dir=logging_dir,
                                      logging_steps=500,
                                      per_device_train_batch_size=training_params.get('batch_size', 2),
                                      per_device_eval_batch_size=6 if args.task == 'img_seg' else 2,
                                      eval_accumulation_steps=10 if args.task == 'img_seg' else None,
                                      num_train_epochs=training_params.get('num_train_epochs', 3),
                                      weight_decay=training_params.get('weight_decay', 1e-2),
                                      lr_scheduler_type='linear',
                                      dataloader_num_workers=1,
                                      learning_rate=learning_rate,
                                      save_total_limit=1,
                                      metric_for_best_model=args.metric_name,
                                      load_best_model_at_end=True,
                                      greater_is_better=True,
                                      disable_tqdm=True,
                                      optim='adamw_torch',
                                      seed=1024,
                                      use_mps_device=device == 'mps',
                                      no_cuda=no_cuda,
                                      remove_unused_columns=False,
                                      bf16=bf16,
                                      fp16=fp16)

    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=data_content['collator'],
                      tokenizer=data_content['tokenizer'],
                      train_dataset=torch.utils.data.Subset(data_content[send_tag], sample_idxs)
                      if sample_idxs else data_content[send_tag],
                      eval_dataset=torch.utils.data.Subset(data_content['val'], val_sample_idxs),
                      compute_metrics=compute_metrics)
    trainer.val_sample_idxs = val_sample_idxs

    return trainer
