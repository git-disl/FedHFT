# FedHFT: Efficient Federated Finetuning with Heterogeneous Edge Clients

This repository contains the code for: "FedHFT: Efficient Federated Finetuning with Heterogeneous Edge Clients" (IEEE CogMI 2025).

Paper link: https://arxiv.org/abs/2510.14054

Example usage:

`python main.py --data $DATASET --arch $ARCH --use-valid --device $DEVICE --num_clusters $NUM_CLUSTERS --vertical_scale_ratios $MASK_RATIO` 

You can check all rum arguments, default values and possible choices in `args.py`.

To experiment with new datasets, check `data_tools.dataloader` and add the data preparation utility function which will be called by `prepare_datasets`. 

To experiment with new models, extend the model and tokenizer dispatchers in `main.py`.

Federated finetuning logic is implemented in `fed.py`.
