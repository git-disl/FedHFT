import argparse
import datetime
import os


def modify_args(args):
    if args.device == 'cuda' and args.gpu_idx:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx

    args.datetime = format(str(datetime.datetime.now()))

    if args.task == 'glue':
        args.metric_name = "pearson" if args.data == "stsb" else "matthews_correlation" if args.data == "cola" else "accuracy"
    elif args.task == 'qa':
        args.metric_name = 'f1'
    else:
        raise NotImplementedError

    args.opt_class = 'adamw'
    args.num_classes = 2
    if args.data == 'ag_news':
        args.num_classes = 4

    args.final_eval_split = 'val'
    if args.use_valid:
        args.splits = ['train', 'val', 'test']
    else:
        args.splits = ['train', 'val']

    if not hasattr(args, "save_path") or args.save_path is None:
        args.save_path = f"outputs/{args.arch}_{args.evalmode}_{args.data}_{format(str(datetime.datetime.now()).replace(' ', '_'))}_{args.num_clients}_{args.num_rounds}_{args.sample_rate}_{args.alpha}"

    return args


model_names = ['bert-base-uncased', 'bert-large-uncased']

arg_parser = argparse.ArgumentParser(
    description='Image classification PK main script')

exp_group = arg_parser.add_argument_group('exp', 'experiment setting')
exp_group.add_argument('--save_path', default=None,
                       type=str, metavar='SAVE',
                       help='path to the experiment logging directory')
exp_group.add_argument('--resume', action='store_true',
                       help='path to latest checkpoint (default: none)')
exp_group.add_argument('--evalmode', default=None,
                       choices=['local', 'global'],
                       help='which mode to evaluate')
exp_group.add_argument('--evaluate_from', default=None, type=str, metavar='PATH',
                       help='path to saved checkpoint (default: none)')
exp_group.add_argument('--print-freq', '-p', default=10, type=int,
                       metavar='N', help='print frequency (default: 100)')
exp_group.add_argument('--seed', default=66, type=int,
                       help='random seed')
exp_group.add_argument('--gpu_idx', default=None, type=str, help='Index of available GPU')
exp_group.add_argument('--device', default='cuda', type=str, choices=['cpu', 'cuda', 'mps'], help='Device type')
exp_group.add_argument('--comp_device', default='cuda', type=str, choices=['cpu', 'cuda', 'mps'], help='Device type')
exp_group.add_argument('--eval_rate', default=1, type=float, help='eval x ratio of val set')
exp_group.add_argument('--eval_period', default=1, type=float, help='eval per x round')

# dataset related
data_group = arg_parser.add_argument_group('data', 'dataset setting')
data_group.add_argument('--task', default='glue')
data_group.add_argument('--data', metavar='D', default='sst2', help='data to work on')
data_group.add_argument('--data-root', metavar='DIR', default='data',
                        help='path to dataset (default: data)')
data_group.add_argument('--use-valid', action='store_true',
                        help='use validation set or not')
data_group.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')
data_group.add_argument('-jj', '--num_fed_workers', default=1, type=int, metavar='N',
                        help='number of fl workers (default: 1)')
# model arch related
arch_group = arg_parser.add_argument_group('arch', 'model architecture setting')
arch_group.add_argument('--arch', '-a', metavar='ARCH', default='bert-base-uncased',
                        type=str, choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: bert)')

# training related
optim_group = arg_parser.add_argument_group('optimization', 'optimization setting')

optim_group.add_argument('--start_round', default=0, type=int, metavar='N',
                         help='manual round number (useful on restarts)')
optim_group.add_argument('-b', '--batch-size', type=int, help='mini-batch size')

# FL related
fl_group = arg_parser.add_argument_group('fl', 'FL setting')
fl_group.add_argument('--vertical_scale_ratios', type=float, nargs='*', default=[0.5],
                      help='model split ratio vertically for each complexity level')
fl_group.add_argument('--client_split_ratios', type=float, nargs='*', default=[1],
                      help='client ratio at each complexity level')
fl_group.add_argument('--num_rounds', type=int, default=20,
                      help='number of rounds')
fl_group.add_argument('--num_warmup_rounds', type=int, default=5,
                      help='number of rounds')
fl_group.add_argument('--num_clients', type=int, default=50,
                      help='number of clients')
fl_group.add_argument('--num_clusters', type=int, default=1,
                      help='number of clusters')
fl_group.add_argument('--sample_rate', type=float, default=1,
                      help='client sample rate')
fl_group.add_argument('--alpha', type=int, default=10,
                      help='data nonIID alpha')

# compression related
comp_group = arg_parser.add_argument_group('comp', 'compression setting')
comp_group.add_argument('--comp_method', '-cm', default='af', type=str,
                        choices=['f', 'af'],
                        help='f:finetune, q:quantize, a:adapter')
comp_group.add_argument('--quant_method', '-qm', default='bf16', type=str, choices=['bf16', 'fp16', 'fp32'],
                        help='quantization technique')
comp_group.add_argument('--adapter_method', '-am', default='lora', type=str, choices=['lora'], help='adapter technique')