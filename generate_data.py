import torch_geometric

import torch_sparse
import torch
import util
from math import floor, ceil

from torch_sparse.tensor import SparseTensor
from tqdm import tqdm
import code

from sklearn.metrics import pairwise_kernels

import wandb

from eval import *

from timeit import default_timer as timer

import argparse


from types import SimpleNamespace


torch_geometric.seed_everything(0)


configs = {
    'facebook':{
        'num_partitions': 1,
        'partition_eval': False,
    },
    'gplus':{
        'num_partitions': 1,
        'partition_eval': False,
    },
    
}



for dataset in configs.keys():
    print(f">>> [PROCESSING DATASET]: {dataset}")
    args = SimpleNamespace(**configs[dataset])

    data, pyg_dataset = util.load_dataset(dataset)

    data.edge_attr = torch.ones([data.edge_index.shape[1], 1], dtype=int)

    if args.num_partitions == 1:
        print('>>> Generating full sampling + full evaluation')
        split_edge = util.split_dataset(data)
    elif not(args.partition_eval):
        print(">>> Generating Partition sampling + Full evaluation")
        split_edge_aux = util.split_dataset_partition_sampling(data, n_partitions=args.num_partitions)
        split_edge = util.split_dataset(data)
        split_edge['train']['edge_neg'] = split_edge_aux['train']['edge_neg']
    else:
        print(">>> Generating Partition sampling + partition evaluation")
        split_edge = util.split_dataset_partition_sampling(data, n_partitions=args.num_partitions)
        
    # num_test_pos_pairs = split_edge['valid']['edge'].shape[0]
    # split_edge['biased_valid'] = dict()
    # split_edge['biased_valid']['edge_pos'] = split_edge['valid']['edge']
    # split_edge['biased_valid']['edge_neg'] = split_edge['valid']['edge_neg'][torch.randint(0, split_edge['valid']['edge_neg'].shape[0], (num_test_pos_pairs,)), :]

    # num_test_pos_pairs = split_edge['test']['edge'].shape[0]
    # split_edge['biased_test'] = dict()
    # split_edge['biased_test']['edge'] = split_edge['test']['edge']
    # split_edge['biased_test']['edge_neg'] = split_edge['test']['edge_neg'][torch.randint(0, split_edge['test']['edge_neg'].shape[0], (num_test_pos_pairs,)), :]
    
    

    if not os.path.isdir('./processed_splits'):
        os.mkdir('./processed_splits')
    torch.save(split_edge, f'./processed_splits/{dataset}_{args.num_partitions}_{args.partition_eval}.pth')

