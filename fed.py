import copy
import os
import pickle as pkl

import numpy as np
import torch
import torch.multiprocessing as mp
from peft import get_peft_model_state_dict
from sklearn.mixture import GaussianMixture

from compression.adapter import inject_adapter
from train import prepare_traced_trainer
from utils.utils import save_checkpoint

mp.set_start_method('spawn', force=True)


class Federator:
    def __init__(self, global_model, args, client_groups=[]):
        self.global_model = global_model

        self.vertical_scale_ratios = args.vertical_scale_ratios
        self.client_split_ratios = args.client_split_ratios

        assert len(self.vertical_scale_ratios) == len(self.client_split_ratios)

        self.num_rounds = args.num_rounds
        self.num_clients = args.num_clients
        self.sample_rate = args.sample_rate
        self.alpha = args.alpha
        self.num_levels = len(self.vertical_scale_ratios)
        self.client_groups = client_groups

        self.num_clusters = args.num_clusters
        self.global_grad_dicts = [{}] * self.num_clusters
        self.lora_dicts = [{}] * self.num_clusters
        self.clusters = np.ones((self.num_clients, self.num_clusters)) / self.num_clusters
        self.gm = None

        self.device = args.device

    def fed_train(self, args, config, data_content, user_groups, batch_size, train_params):

        scores = ['epoch\tval_score']
        best_score, best_round = 0.0, 0

        # pre-assignment of levels to clients (needs to be saved for inference)
        if not self.client_groups:
            client_idxs = np.arange(self.num_clients)
            np.random.seed(args.seed)
            shuffled_client_idxs = np.random.permutation(client_idxs)
            client_groups = []
            s = 0
            for ratio in self.client_split_ratios:
                e = s + int(len(shuffled_client_idxs) * ratio)
                client_groups.append(shuffled_client_idxs[s: e])
                s = e
            self.client_groups = client_groups

            with open(os.path.join(args.save_path, 'client_groups.pkl'), 'wb') as f:
                pkl.dump(self.client_groups, f)

        for round_idx in range(args.start_round, self.num_rounds):

            print(f'\n | Global Training Round : {round_idx + 1} |\n')

            val_score = \
                self.execute_round(args, config, data_content, user_groups, batch_size,
                                   train_params, round_idx)

            if (round_idx + 1) % args.eval_period == 0:
                scores.append(('{}' + '\t{:.4f}').format(round_idx, val_score))

                is_best = val_score > best_score
                if is_best:
                    best_score = val_score
                    best_round = round_idx
                    print('Best val_score {}'.format(best_score))
                else:
                    print('This val_score {}'.format(val_score))

                model_filename = 'checkpoint_%03d.pth.tar' % round_idx
                save_checkpoint({
                    'round': round_idx,
                    'arch': args.arch,
                    'state_dict': self.global_model.state_dict(),
                    'best_score': best_score,
                }, args, is_best, model_filename, scores, data_content['stats_df'])
            else:
                pass

        return best_score, best_round

    def get_level(self, client_idx):
        # Return the complexity level of given client, starts with 0
        try:
            level = np.where([client_idx in c for c in self.client_groups])[0][0]
        except:
            # client will be skipped
            level = -1

        return level
    
    def compute_masks(self, lora_weight_dict, scale):
        masks = {}
        keys = sorted(list(set(['.'.join(x.split('.')[:-2]) for x in lora_weight_dict.keys() if 'lora' in x])))
        for key in keys:
            dw = lora_weight_dict[key + '.lora_B.weight'] @ lora_weight_dict[key + '.lora_A.weight']
            vals = (dw * dw).sum(dim=1)
            thr = vals.sort()[0][int(len(vals) * (1-scale))]
            masks[key] = vals >= thr 
        return masks

    def execute_round(self, args, config, data_content, user_groups, batch_size, train_params, round_idx):
        self.global_model.train()
        m = max(int(self.sample_rate * self.num_clients), 1)

        if self.sample_rate == 1:
            client_idxs = np.arange(self.num_clients)
        else:
            client_idxs = np.random.choice(range(self.num_clients), m, replace=False)

        levels = [self.get_level(client_idx) for client_idx in client_idxs]
        scales = [self.vertical_scale_ratios[level] for level in levels]
        client_data_contents = [{'train': data_content['train'].select(user_groups[0][client_idxs[i]]),
                                 'val': data_content['val'].select(user_groups[1][client_idxs[i]]),
                                 'tokenizer': data_content['tokenizer'],
                                 'collator': data_content['collator']} for i in range(len(client_idxs))]

        pool_args = [args, config, round_idx]
        lora_weights = []
        val_scores = []
        val_lengths = []

        for i, client_idx in enumerate(client_idxs):
            local_model = self.get_local_model(client_idx, args, config)
            client_args = pool_args + [client_idx, local_model, client_data_contents[i]]
            lora_weight_dict, trainer_state, val_score = execute_client_round(client_args)
            if scales[i] != 1:
                masks = self.compute_masks(lora_weight_dict, scales[i])
            else:
                masks = None
            lora_weights.append([lora_weight_dict, masks])
            val_scores.append(val_score)
            val_lengths.append(len(client_args[-1]['val']))
            print(f'Client {i + 1}/{len(client_idxs)} with ID: {client_idxs[i]} and probs: {self.clusters[client_idx]} finished')

        val_score_final = sum([val_scores[i] * val_lengths[i] for i in range(self.num_clients)]) / sum(val_lengths)

        # Update the global model
        # self.global_model = inject_adapter(self.global_model, args, config)
        grad_dicts, classifier_dicts = self.average_weights(lora_weights, client_idxs)

        for j in range(self.num_clusters):
            for key in grad_dicts[j].keys():
                rank = lora_weights[0][0][key[:-5] + '.lora_B.weight'].shape[-1]
                if key in self.global_grad_dicts[j].keys():
                    self.global_grad_dicts[j][key] += grad_dicts[j][key]
                else:
                    self.global_grad_dicts[j][key] = grad_dicts[j][key]
                b_, s, a_ = torch.svd_lowrank(self.global_grad_dicts[j][key], q=rank)
                b_ = b_ @ torch.diag(s)
                self.lora_dicts[j][key[:-5] + f'.lora_B.default.weight'] = b_
                self.lora_dicts[j][key[:-5] + f'.lora_A.default.weight'] = a_.t()

            if classifier_dicts[0]:
                key = 'base_model.model.classifier'
                self.lora_dicts[j][key + f'.modules_to_save.default.weight'] = classifier_dicts[j][key + '.weight']
                self.lora_dicts[j][key + f'.modules_to_save.default.bias'] = classifier_dicts[j][key + '.bias']

        del grad_dicts
        del classifier_dicts

        if round_idx >= args.num_warmup_rounds and self.num_clusters > 1:
            self.update_clusters(lora_weights, client_idxs)

        return val_score_final

    def update_clusters(self, lora_weights, client_idxs):
        if any(['classifier' in k for k in lora_weights[0][0].keys()]):
            weights = np.stack(
                [l[0]['base_model.model.classifier.weight'].flatten().cpu().numpy() for l in lora_weights])
        else:
            weights = np.stack(
                [torch.concat([l[0][k].flatten() for k in lora_weights[0][0].keys()]).cpu().numpy() for l in lora_weights])

        if len(lora_weights) == 1:
            probs = [1.]
        else:
            self.gm = GaussianMixture(n_components=self.num_clusters, random_state=0).fit(weights)
            probs = self.gm.predict_proba(weights)
        self.clusters[client_idxs] = probs

    def average_weights(self, lora_weights, client_idxs):

        grad_dicts = [{}] * self.num_clusters
        classifier_dicts = [{}] * self.num_clusters
        keys = sorted(
            list(set(['.'.join(x.split('.')[:-2]) for x in lora_weights[0][0].keys() if 'lora' in x])) +
            [x for x in lora_weights[0][0].keys() if 'lora' not in x])
        for key in keys:
            if 'classifier' not in key:
                weight_shape = self.global_model.state_dict()[key[17:] + '.weight'].shape
                tmp_list = [torch.zeros(weight_shape, device=self.device)] * self.num_clusters
                count = torch.zeros(weight_shape, device=self.device)

                for i in range(len(lora_weights)):
                    masks = lora_weights[i][1]
                    if masks is None:
                        dw = lora_weights[i][0][key + '.lora_B.weight'] @ lora_weights[i][0][key + '.lora_A.weight']
                        for j in range(self.num_clusters):
                            tmp_list[j] += dw * self.clusters[client_idxs[i], j]
                        count += 1
                    else:
                        mask = masks[key]
                        dw = lora_weights[i][0][key + '.lora_B.weight'] @ lora_weights[i][0][key + '.lora_A.weight']
                        for j in range(self.num_clusters):
                            tmp_list[j][mask] += dw[mask] * self.clusters[client_idxs[i], j]
                        count[mask] += 1
                count[count == 0] = 1

                for j in range(self.num_clusters):
                    tmp_list[j] = tmp_list[j] / count
                    grad_dicts[j][key + '.grad'] = tmp_list[j]

            else:
                weight_shape = self.global_model.state_dict()[key[17:]].shape
                tmp_list = [torch.zeros(weight_shape, device=self.device)] * self.num_clusters

                count = 0
                for i in range(len(lora_weights)):
                    dw = lora_weights[i][0][key]
                    for j in range(self.num_clusters):
                        tmp_list[j] += dw * self.clusters[client_idxs[i], j]
                    count += 1
                for j in range(self.num_clusters):
                    tmp_list[j] = tmp_list[j] / count
                    classifier_dicts[j][key] = tmp_list[j]

        return grad_dicts, classifier_dicts

    def merge_lora_weights(self, cluster_weights):
        if self.num_clusters == 1:
            return self.lora_dicts[0]
        
        merged_lora_dict = {}
        keys = sorted(
            list(set(['.'.join(x.split('.')[:-3]) for x in self.lora_dicts[0].keys() if 'lora' in x])) +
            [x for x in self.lora_dicts[0].keys() if 'lora' not in x])
        for k in keys:
            if 'classifier' not in k:
                rank = self.lora_dicts[0][k + '.lora_B.default.weight'].shape[-1]
                dw = torch.sum(torch.stack([cluster_weights[cluster_idx] * self.global_grad_dicts[cluster_idx][k + '.grad']
                                                            for cluster_idx in range(self.num_clusters)]),
                                                dim=0)
                b_, s, a_ = torch.svd_lowrank(dw, q=rank)
                b_ = b_ @ torch.diag(s)
                merged_lora_dict[k + f'.lora_B.default.weight'] = b_
                merged_lora_dict[k + f'.lora_A.default.weight'] = a_.t()
            else:
                merged_lora_dict[k] = torch.sum(torch.stack([cluster_weights[cluster_idx] * lora_dict[k]
                                                            for cluster_idx, lora_dict in enumerate(self.lora_dicts)]),
                                                dim=0)
        return merged_lora_dict

    def get_local_model(self, client_idx, args, config, inject_lora_for_train=True):
        model = copy.deepcopy(self.global_model)
        if self.lora_dicts[0]:
            model = inject_adapter(model, args, config)
            model.load_state_dict(self.merge_lora_weights(self.clusters[client_idx]), strict=False)
            model.merge_and_unload()
            model = model.base_model.model

        if inject_lora_for_train:
            model = inject_adapter(model, args, config)

        return model


def execute_client_round(client_args):
    args, config, round_idx, client_idx, local_model, data_content = client_args

    if args.device == 'cuda':
        local_model = local_model.cuda()

    training_params = config.get_init_training_params(args.arch, args.data)
    trainer = prepare_traced_trainer(local_model, args, data_content, training_params, for_eval_flag=True,
                                     tag=f'finetune_{round_idx}_{client_idx}', eval_rate=1)
    trainer.train()
    lora_weight_dict = get_peft_model_state_dict(local_model)
    classifier_dict = {}
        
    tag = f'validate_{round_idx+1}_{client_idx+1}'
    val_output = predict(local_model, args, data_content, tag=tag)
    val_score = val_output.metrics[f'{tag}_{args.metric_name}']

    return lora_weight_dict | classifier_dict, trainer.state, val_score


def predict(model, args, data_content, tag='default'):
    trainer = prepare_traced_trainer(model.to(args.device), args, data_content, {}, for_train_flag=False, tag=tag,
                                     eval_rate=args.eval_rate)
    output = trainer.predict(data_content[send_tag], metric_key_prefix=tag)
    print(f'Metric: {output.metrics}')
    return output
