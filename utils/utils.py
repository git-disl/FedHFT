import glob
import os
import pickle
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from transformers.trainer_pt_utils import get_parameter_names


def save_checkpoint(state, args, is_best, filename, result, stats_df):
    print(args)
    result_filename = os.path.join(args.save_path, 'scores.tsv')
    stats_df_filename = os.path.join(args.save_path, 'stats_df.tsv')
    model_dir = os.path.join(args.save_path, 'save_models')
    model_filename = os.path.join(model_dir, filename)
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print("=> saving checkpoint '{}'".format(model_filename))

    with open(result_filename, 'a') as f:
        print(result[-1], file=f)

    with open(stats_df_filename, 'w') as f:
        print(stats_df, file=f)

    print("=> saved checkpoint '{}'".format(model_filename))
    return


def load_checkpoint(args, load_best=True):
    model_dir = os.path.join(args.save_path, 'save_models')
    if load_best:
        model_filename = os.path.join(model_dir, 'model_best.pth.tar')
    else:
        model_filename = glob.glob(os.path.join(model_dir, 'checkpoint*'))[0]

    if os.path.exists(model_filename):
        print("=> loading checkpoint '{}'".format(model_filename))
        state = torch.load(model_filename)
        print("=> loaded checkpoint '{}'".format(model_filename))
    else:
        return None

    return state


def save_user_groups(args, user_groups):
    user_groups_filename = os.path.join(args.save_path, 'user_groups.pkl')
    if not os.path.exists(user_groups_filename):
        with open(user_groups_filename, 'wb') as fout:
            pickle.dump(user_groups, fout)


def load_user_groups(args):
    user_groups_filename = os.path.join(args.save_path, 'user_groups.pkl')

    if os.path.exists(user_groups_filename):
        with open(user_groups_filename, 'rb') as fin:
            user_groups = pickle.load(fin)
    else:
        user_groups = None

    return user_groups


def load_state_dict(args, model):
    state_dict = torch.load(args.evaluate_from)['state_dict']
    model.load_state_dict(state_dict)

