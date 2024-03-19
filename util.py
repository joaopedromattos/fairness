import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from torch_geometric.datasets import Planetoid, Amazon, KarateClub
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.utils import negative_sampling, add_self_loops, train_test_split_edges, k_hop_subgraph
from torch_geometric.transforms import RandomLinkSplit
from tqdm import tqdm
import argparse 
from ast import literal_eval
import code
import torch_geometric
import os
from torch_sparse.tensor import SparseTensor
from itertools import combinations
from tqdm import tqdm
import scipy.sparse as ssp
import datetime

import time


def create_timestamp_folder(directory, dataset):
    # Get the current date and time
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d_%H-%M")

    # Create the folder name with the timestamp
    folder_name = os.path.join(directory, timestamp + f'_{dataset}')

    # Create the folder
    try:
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully.")
    except OSError as e:
        print(f"Error creating folder: {e}")

    return folder_name


def compute_edges(split_edge, heart=False):
    """
    Compute the train, valid, and test edges based on edge split.

    :param split_edge: edge split.
    :return: train_edges_pos, train_edges_neg, valid_edges, valid_true, test_edges, test_true
    """

    # Train edges.
    train_edges_pos = split_edge['train']['edge']
    train_edges_pos = train_edges_pos[train_edges_pos[:, 0] < train_edges_pos[:, 1]]  # Only include upper triangle.
    train_edges_neg = split_edge['train']['edge_neg']

    # code.interact(local=locals())
    # Valid edges.
    valid_edges = torch.vstack([split_edge['valid']['edge'], split_edge['valid']['edge_neg']])
    valid_true = torch.cat([torch.ones(split_edge['valid']['edge'].shape[0], dtype=int), torch.zeros(split_edge['valid']['edge_neg'].shape[0], dtype=int)])

    if not heart:
        index = torch.randperm(valid_edges.shape[0])  # Shuffle edges for expected values of precision@k for ties.
        valid_edges = valid_edges[index]
        valid_true = valid_true[index]

    # Test edges.
    test_edges = torch.vstack([split_edge['test']['edge'], split_edge['test']['edge_neg']])
    test_true = torch.cat([torch.ones(split_edge['test']['edge'].shape[0], dtype=int), torch.zeros(split_edge['test']['edge_neg'].shape[0], dtype=int)])

    if not heart:
        index = torch.randperm(test_edges.shape[0])
        test_edges = test_edges[index]
        test_true = test_true[index]

    return train_edges_pos, train_edges_neg, valid_edges, valid_true, test_edges, test_true


def set_random_seed(random_seed):
    """
    Set the random seed.
    :param random_seed: Seed to be set.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def split_dataset(data, valid_ratio=0.05, test_ratio=0.1, random_seed=0, heart_splits=None):
    """
    Split the edges/nonedges for biased training, full training, (full) validation and (full) testing.

    :param data: PyG dataset data.
    :param valid_ratio: ratio of validation edges.
    :param test_ratio: ratio of test edges.
    :param random_seed: random seed for the split.
    :return: edge splits
    """
    
    set_random_seed(random_seed)
    n = data.num_nodes

    if heart_splits:
        from types import SimpleNamespace
        split_data = SimpleNamespace(train_pos_edge_index=heart_splits['train']['edge'].t(), val_pos_edge_index=heart_splits['valid']['edge'].t(), test_pos_edge_index=heart_splits['test']['edge'].t(), num_nodes=n)

    else:
        split_data = train_test_split_edges(data, valid_ratio, test_ratio)


    split_edge = {'biased_train': {}, 'valid': {}, 'test': {}, 'train': {}}

    # Biased training with negative sampling.
    split_edge['biased_train']['edge'] = split_data.train_pos_edge_index.t()
    edge_index, _ = add_self_loops(split_data.train_pos_edge_index)  # To avoid negative sampling of self loops.
    split_data.train_neg_edge_index = negative_sampling(
        edge_index, num_nodes=split_data.num_nodes,
        num_neg_samples=split_data.train_pos_edge_index.size(1))
    split_edge['biased_train']['edge_neg'] = split_data.train_neg_edge_index.t()

    # Full training with all negative pairs in the training graph (including validation and testing positive edges).
    split_edge['train']['edge'] = split_data.train_pos_edge_index.t()
    train_edge_neg_mask = torch.ones((n, n), dtype=bool)
    train_edge_neg_mask[tuple(split_edge['train']['edge'].t().tolist())] = False
    train_edge_neg_mask = torch.triu(train_edge_neg_mask, 1)
    split_edge['train']['edge_neg'] = torch.nonzero(train_edge_neg_mask)

    # Full validation with all negative pairs in the training graph (including testing positive edges, excluding validation positive edges).
    split_edge['valid']['edge'] = split_data.val_pos_edge_index.t()
    valid_edge_neg_mask = train_edge_neg_mask.clone()
    valid_edge_neg_mask[tuple(split_edge['valid']['edge'].t().tolist())] = False
    split_edge['valid']['edge_neg'] = torch.nonzero(valid_edge_neg_mask)

    # Full testing with all negative pairs in the training graph (excluding validation and testing positive edges).
    split_edge['test']['edge'] = split_data.test_pos_edge_index.t()
    test_edge_neg_mask = valid_edge_neg_mask.clone()
    test_edge_neg_mask[tuple(split_edge['test']['edge'].t().tolist())] = False
    split_edge['test']['edge_neg'] = torch.nonzero(test_edge_neg_mask)

    return split_edge



def split_dataset_partition_sampling(data, n_partitions=3, valid_ratio=0.05, test_ratio=0.1, random_seed=0, heart_splits=None):
    """
    Split the edges/nonedges for Cluster-importance sampling.
    As described in the paper, the idea is to sample negative pairs
    from within graph partitions.

    :param data: PyG dataset data.
    :param valid_ratio: ratio of validation edges.
    :param test_ratio: ratio of test edges.
    :param random_seed: random seed for the split.
    :return: edge splits
    """

    set_random_seed(random_seed)

    adj = SparseTensor.from_edge_index(data.edge_index, edge_attr=torch.ones(size=(data.edge_index.shape[1],)))
        
    partitions, part_idx, part_nodes = adj.partition(n_partitions)

    # partitions, part_idx, part_nodes = torch.load("./7000_partitions_vessel.pth")

    neg_edge_index = None

    for part_start, part_end in tqdm(list(zip(part_idx, part_idx[1:-1]))):
        cur_partition_nodes = part_nodes[part_start:part_end + 1].tolist()

        all_possible_pairs = torch.tensor(list(combinations(cur_partition_nodes, 2)))

        if neg_edge_index is None:
            neg_edge_index = all_possible_pairs
        else:
            neg_edge_index = torch.cat([neg_edge_index, all_possible_pairs])

    
    neg_edge_index = neg_edge_index.t()

    set_random_seed(random_seed)
    n = data.num_nodes

    if heart_splits:
        from types import SimpleNamespace
        train_data = SimpleNamespace(edge_label_index=heart_splits['train']['edge'].t())
        val_data = SimpleNamespace(edge_label_index=heart_splits['valid']['edge'].t()) 
        test_data = SimpleNamespace(edge_label_index=heart_splits['test']['edge'].t())

    else:
        transform = RandomLinkSplit(add_negative_train_samples=False, num_val=valid_ratio, num_test=test_ratio)
        train_data, val_data, test_data = transform(data)

    # code.interact(local=locals())

    # split_data = train_test_split_edges(data, valid_ratio, test_ratio)
    split_edge = {'biased_train': {}, 'valid': {}, 'test': {}, 'train': {}}

    # Full training with all negative pairs in the training graph (including validation and testing positive edges).
    split_edge['train']['edge'] = train_data.edge_label_index.t()
    all_edges_train = torch.cat([data.edge_index, train_data.edge_label_index], dim=1)
    train_edge_label_mask = SparseTensor.from_edge_index(all_edges_train, torch.full(size=(all_edges_train.shape[1],), fill_value=-1), adj.sizes()).coalesce(reduce='max')
    neg_edge_index_mask_train = SparseTensor.from_edge_index(neg_edge_index, torch.ones(size=(neg_edge_index.shape[1],)), adj.sizes())
    row, col, mask = (neg_edge_index_mask_train + train_edge_label_mask).coo()
    split_edge['train']['edge_neg'] = torch.vstack([row[mask == 1], col[mask == 1]]).t()


    # Full validation with all negative pairs in the training graph (including testing positive edges, excluding validation positive edges).
    split_edge['valid']['edge'] = val_data.edge_label_index.t()
    all_edges_val = torch.cat([data.edge_index, train_data.edge_label_index, val_data.edge_label_index], dim=1)
    val_edge_label_mask = SparseTensor.from_edge_index(all_edges_val, torch.full(size=(all_edges_val.shape[1],), fill_value=-1), adj.sizes()).coalesce(reduce='max')
    neg_edge_index_mask_val = SparseTensor.from_edge_index(neg_edge_index, torch.ones(size=(neg_edge_index.shape[1],)), adj.sizes())
    row, col, mask = (neg_edge_index_mask_val + val_edge_label_mask).coo()
    split_edge['valid']['edge_neg'] = torch.vstack([row[mask == 1], col[mask == 1]]).t()


    # Full testing with all negative pairs in the training graph (excluding validation and testing positive edges).
    split_edge['test']['edge'] = test_data.edge_label_index.t()
    all_edges_test = torch.cat([data.edge_index, train_data.edge_label_index, val_data.edge_label_index, test_data.edge_label_index], dim=1)
    test_edge_label_mask = SparseTensor.from_edge_index(all_edges_test, torch.full(size=(all_edges_test.shape[1],), fill_value=-1), adj.sizes()).coalesce(reduce='max')
    neg_edge_index_mask_test = SparseTensor.from_edge_index(neg_edge_index, torch.ones(size=(neg_edge_index.shape[1],)), adj.sizes())
    
    del row, col, mask, neg_edge_index_mask_val, val_edge_label_mask, all_edges_val, neg_edge_index_mask_train
    # import code
    # code.interact(local=locals())
    row, col, mask = (neg_edge_index_mask_test + test_edge_label_mask).coo()
    split_edge['test']['edge_neg'] = torch.vstack([row[mask == 1], col[mask == 1]]).t()

    print(">>>> finished sampling neg samples")
    # code.interact(local=locals())

    return split_edge



# def (dataset:str, heart_files_path:str=''):

#     assert os.path.exists(heart_files_path), f"'{heart_files_path}' -- Path to files does not exist, please check"



#     val_neg_samples = torch.from_numpy(np.load(os.path.join(heart_files_path, 'heart_valid_samples.npy')))
#     test_neg_samples = torch.from_numpy(np.load(os.path.join(heart_files_path, 'heart_test_samples.npy')))

#     if ('ogbl' not in dataset):



def heart_read_data(data, data_name, dir_path, filename, partitions=None):

    print(">>> Reading HeaRT splits...")

    data_name = data_name

    node_set = set()
    train_pos, valid_pos, test_pos = [], [], []
    train_neg, valid_neg, test_neg = [], [], []

    for split in ['train', 'test', 'valid']:

        
        path = dir_path+ '/{}/{}_pos.txt'.format(data_name, split)
      
        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)
            
            node_set.add(sub)
            node_set.add(obj)
            
            if sub == obj:
                continue

            if split == 'train': 
                train_pos.append((sub, obj))
                

            if split == 'valid': valid_pos.append((sub, obj))  
            if split == 'test': test_pos.append((sub, obj))
    
    num_nodes = len(node_set)
    print('the number of nodes in ' + data_name + ' is: ', num_nodes)

    

    train_edge = torch.transpose(torch.tensor(train_pos), 1, 0)
    edge_index = torch.cat((train_edge,  train_edge[[1,0]]), dim=1)
    edge_weight = torch.ones(edge_index.size(1))

    with open(f'{dir_path}/{data_name}/heart_valid_{filename}', "rb") as f:
        valid_neg = np.load(f)
        valid_neg = torch.from_numpy(valid_neg)
    with open(f'{dir_path}/{data_name}/heart_test_{filename}', "rb") as f:
        test_neg = np.load(f)
        test_neg = torch.from_numpy(test_neg)


    A = ssp.csr_matrix((edge_weight.view(-1), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes)) 

    adj = SparseTensor.from_edge_index(edge_index, edge_weight, [num_nodes, num_nodes])
          

    train_pos_tensor = torch.tensor(train_pos)

    valid_pos = torch.tensor(valid_pos)
    
    test_pos =  torch.tensor(test_pos)
    

    idx = torch.randperm(train_pos_tensor.size(0))
    idx = idx[:valid_pos.size(0)]
    train_val = train_pos_tensor[idx]


    feature_embeddings = torch.load(dir_path+ '/{}/{}'.format(data_name, 'gnn_feature'))
    feature_embeddings = feature_embeddings['entity_embedding']

    splits = {}
    splits['train'] = dict()
    splits['valid'] = dict()
    splits['test'] = dict()
    # splits['adj'] = adj
    splits['train']['edge'] = train_pos_tensor
    splits['train']['train_val'] = train_val

    splits['valid']['edge'] = valid_pos
    splits['valid']['edge_neg'] = valid_neg
    splits['test']['edge'] = test_pos
    splits['test']['edge_neg'] = test_neg

    # code.interact(local=locals())

    # We will use the full training negative samples
    # to train the model, that's why we are re-using the "split_dataset()"
    # function.
    if partitions > 1:
        print(">>> Sampling negative pairs from partition sampling...")
        splits_partition_sampling_full_training = split_dataset_partition_sampling(data, heart_splits=splits, n_partitions=partitions)
        splits['train']['edge_neg'] = splits_partition_sampling_full_training['train']['edge_neg']
        print(">>> ...ok")

    else:
        print(">>> Sampling negative pairs from full training...")
        splits_full_training = split_dataset(data=data, heart_splits=splits)
        splits['train']['edge_neg'] = splits_full_training['train']['edge_neg']
        print(">>> ...ok")


    return splits
        


def split_dataset_ogbl(data, valid_ratio=0.05, test_ratio=0.1, random_seed=0):
    """
    Split the edges/nonedges for biased training, full training, (full) validation and (full) testing.

    :param data: PyG dataset data.
    :param valid_ratio: ratio of validation edges.
    :param test_ratio: ratio of test edges.
    :param random_seed: random seed for the split.
    :return: edge splits
    """

    set_random_seed(random_seed)

    n = data.num_nodes
    transform = RandomLinkSplit()
    train_data, val_data, test_data = transform(data)

    # split_data = train_test_split_edges(data, valid_ratio, test_ratio)
    split_edge = {'biased_train': {}, 'valid': {}, 'test': {}, 'train': {}}

    # Biased training with negative sampling.
    train_label_index = train_data.edge_label.bool()
    split_edge['biased_train']['edge'] = train_data.edge_label_index[:, train_label_index].t()
    split_edge['biased_train']['edge_neg'] = train_data.edge_label_index[:, ~train_label_index].t()

    # Full training with all negative pairs in the training graph (including validation and testing positive edges).
    split_edge['train']['edge'] = train_data.edge_label_index[:, train_label_index].t()
    # train_edge_neg_mask = torch.ones((n, n), dtype=bool)
    # train_edge_neg_mask[tuple(split_edge['train']['edge'].t().tolist())] = False
    # train_edge_neg_mask = torch.triu(train_edge_neg_mask, 1)
    # split_edge['train']['edge_neg'] = torch.tensor([[], []]).t()
    split_edge['train']['edge_neg'] = train_data.edge_label_index[:, ~train_label_index].t()

    # Full validation with all negative pairs in the training graph (including testing positive edges, excluding validation positive edges).
    val_label_index = val_data.edge_label.bool()
    split_edge['valid']['edge'] = val_data.edge_label_index[:, val_label_index].t()
    # valid_edge_neg_mask = train_edge_neg_mask.clone()
    # valid_edge_neg_mask[tuple(split_edge['valid']['edge'].t().tolist())] = False
    # split_edge['valid']['edge_neg'] = torch.tensor([[], []]).t()
    split_edge['valid']['edge_neg'] = val_data.edge_label_index[:, ~val_label_index].t()

    # Full testing with all negative pairs in the training graph (excluding validation and testing positive edges).
    test_label_index = test_data.edge_label.bool()
    split_edge['test']['edge'] = test_data.edge_label_index[:, test_label_index].t()
    # test_edge_neg_mask = valid_edge_neg_mask.clone()
    # test_edge_neg_mask[tuple(split_edge['test']['edge'].t().tolist())] = False
    split_edge['test']['edge_neg'] = test_data.edge_label_index[:, ~test_label_index].t()

    return split_edge



def remove_intercluster(pairs, node_to_partition):
    '''
    Given a list of edges (pairs) in the PyG format and a partition mapper dictionary
    (node_to_partition), removes from the list of pairs the ones that contain nodes
    from different graph partitions
    '''
    

    # Checks, for every pair, if both nodes are in the same
    # partition. If so, then they are possible collisions between
    # the pairs sampled in the clusters and the original validation
    # and test, so we will maintain these pairs.
    maintain_edge = []
    for pair in tqdm(pairs.t(), 'Finding pairs intercluster'):
        i, j = int(pair[0]), int(pair[1])

        i_partition = node_to_partition[i]['node_partition_idx']
        j_partition = node_to_partition[j]['node_partition_idx']

        intracluster = (i_partition == j_partition)

        maintain_edge.append(i_partition if intracluster else -1)
    
    maintain_edge = torch.tensor(maintain_edge)

    # print(f"{sum(maintain_edge != -1)} intraclusters pairs were found...")

    return pairs[:, maintain_edge != -1], maintain_edge



def _remove_intercluster(pairs, node_to_partition):
    '''
    Given a list of edges (pairs) in the PyG format and a partition mapper dictionary
    (node_to_partition), removes from the list of pairs the ones that contain nodes
    from different graph partitions
    '''
    

    # Checks, for every pair, if both nodes are in the same
    # partition. If so, then they are possible collisions between
    # the pairs sampled in the clusters and the original validation
    # and test, so we will maintain these pairs.
    maintain_edge = []
    for pair in tqdm(pairs.t(), 'Finding pairs intercluster'):
        i, j = int(pair[0]), int(pair[1])

        i_partition = node_to_partition[i]['node_partition_idx']
        j_partition = node_to_partition[j]['node_partition_idx']

        intracluster = (i_partition == j_partition)

        maintain_edge.append(i_partition if intracluster else -1)
    
    maintain_edge = torch.tensor(maintain_edge)

    # print(f"{sum(maintain_edge != -1)} intraclusters pairs were found...")

    return pairs[:, maintain_edge != -1], maintain_edge



def remove_dataleak(intracluster_pairs, node_to_partition, splits):
    '''
    Given a list of intracluster_pairs, checks if any of the pairs
    is present in the training pairs contained in 'splits'.

    If a given splits ends up having more positive than negative pairs,
    this function will also rebalance this.
    '''

    for pair in tqdm(intracluster_pairs.t(), desc="Removing dataleaks"):
        i, j = int(pair[0]), int(pair[1])

        partition = node_to_partition[i]['node_partition_idx']

        i_partition = node_to_partition[i]['node_idx']
        j_partition = node_to_partition[j]['node_idx']

        train_edges_neg = splits[partition]['train_edges_neg']

        mask = torch.all(train_edges_neg.t() == torch.tensor([i_partition, j_partition]), dim=1)
        mask += torch.all(train_edges_neg.t() == torch.tensor([j_partition, i_partition]), dim=1)

        mask = ~mask

        splits[partition]['train_edges_neg'] = train_edges_neg[:, mask]


    # Balancing the number of positive pairs
    # in comparisson to the number of negative pairs
    # per split.
    for partition in splits.keys():
        num_neg_pairs = splits[partition]['train_edges_neg'].shape[1]
        num_pos_pairs = splits[partition]['train_edges_pos'].shape[1]

        if (num_pos_pairs > num_neg_pairs):
            splits[partition]['train_edges_pos'] = splits[partition]['train_edges_pos'][:, :num_neg_pairs]

    return splits



def extract_cluster_edge_index(pair, node_to_partition, partition):

    i, j = int(pair[0]), int(pair[1])

    i_partition = node_to_partition[i]['node_partition_idx']
    j_partition = node_to_partition[j]['node_partition_idx']
    
    return (i_partition == partition) and (j_partition == partition)


def extract_cluster_edge_index_for(pairs, node_to_partition):

    converted_pairs = [(node_to_partition[int(pair[0])]['node_idx'], node_to_partition[int(pair[1])]['node_idx']) for pair in pairs.t().tolist()]
        
    return torch.tensor(converted_pairs).t()



def convert_to_partition_index(pairs, node_to_partition):

    converted_pairs = [(node_to_partition[int(pair[0])]['node_idx'], node_to_partition[int(pair[1])]['node_idx']) for pair in pairs.t().tolist()]
        
    return torch.tensor(converted_pairs).t()



def graph_splits(graph_partitions, train_graph, val_graph, test_graph, full_training=True, save_dir=None, ogbl=False):
    '''
    Given the train, validation and test graphs, along with the graph partitions, this function
    organizes our training data in different splits, one per graph partition and removes possible leaks between
    training, validation and test due to the sampling procedure done per partition.
    
    :param graph_partitions: ClusterData object of the graph partitioned.
    :param train_graph: torch_geometric.data.Data object with the training graph as the edge index.
    :param val_graph: torch_geometric.data.Data object with the validation pairs.
    :param test_graph: torch_geometric.data.Data object with the test pairs.
    :param full_training: generates splits using full training (True) or biased training (False).
    :param save_dir: path in which we will cache the splits.
    :param ogbl: in case we are using full-training in ogbl datasets, it samples validation and test negative pairs only within each cluster.
    
    :return: splits, intercluster_splits, node_to_partition
    '''

    n_partitions = len(graph_partitions)

    # If we have cached data, then we will just read it
    if save_dir and os.path.isfile(save_dir + f'splits_{n_partitions}.pt'):
        splits = torch.load(save_dir + f'splits_{n_partitions}.pt')
        intercluster_splits = torch.load(save_dir + f'intercluster_splits_{n_partitions}.pt')
        node_to_partition = torch.load(save_dir + f'node_to_partition_{n_partitions}.pt')

        return splits, intercluster_splits, node_to_partition


    # Stores partition metadata regarding each node of the network.
    node_to_partition = {}
    partition_idx = 0
    node_partition_idx = 0
    for idx, node in enumerate(graph_partitions.perm):
        if idx >= graph_partitions.partptr[partition_idx + 1]:
            partition_idx += 1
            node_partition_idx = 0

        node_to_partition[int(node)] = dict()
        node_to_partition[int(node)]['node_partition_idx'] = partition_idx # In which partition this node is located
        node_to_partition[int(node)]['node_idx'] = node_partition_idx # What is its index in the partition it is located

        node_partition_idx += 1
    
    splits = dict()


    valid_edge_index = val_graph.edge_index
    test_edge_index = test_graph.edge_index

    valid_edges = val_graph.edge_label_index
    valid_true = val_graph.edge_label

    test_edges = test_graph.edge_label_index
    test_true = test_graph.edge_label

    intercluster_mask_val = torch.zeros((val_graph.edge_label_index.shape[1])).bool()
    intercluster_mask_test = torch.zeros((test_graph.edge_label_index.shape[1])).bool()

    intracluster_pairs_val, intracluster_idx_val = remove_intercluster(val_graph.edge_label_index, node_to_partition=node_to_partition)
    intracluster_pairs_test, intracluster_idx_test = remove_intercluster(test_graph.edge_label_index, node_to_partition=node_to_partition)

    intercluster_mask_val = (intracluster_idx_val == - 1)
    intercluster_mask_test = (intracluster_idx_test == -1) 
    
    _, intracluster_edge_idx_val = remove_intercluster(val_graph.edge_index, node_to_partition=node_to_partition)
    _, intracluster_edge_idx_test = remove_intercluster(test_graph.edge_index, node_to_partition=node_to_partition)

    
    for idx, partition in tqdm(enumerate(graph_partitions), desc="Processing graph partitions"):

        subgraph = partition

        train_edge_index = subgraph.edge_index

        splits[idx] = dict()

        splits[idx]['subgraph'] = subgraph

        splits[idx]['train_edge_index'] = train_edge_index


        del subgraph.edge_label, subgraph.edge_label_index

        
        # Resampling subgraph pairs.
        # The idea here is to take advantage of the fact that we have
        # a subgraph computed using METIS to sample more informative pairs
        # than the ones contained in the original training set.
        transform = RandomLinkSplit(is_undirected=False, num_val=0.0, num_test=0.0)
        subgraph_train, subgraph_val, subgraph_test = transform(subgraph)


        num_pos_edges = subgraph_train.edge_label.bool().sum()
        num_neg_edges = (~subgraph_train.edge_label.bool()).sum()

        train_edges_pos = subgraph_train.edge_label_index[:, subgraph_train.edge_label.bool()]

        if full_training:
            n = subgraph.num_nodes
            # code.interact(local=locals())
            train_edge_neg_mask = torch.ones((n, n), dtype=bool)
            train_edge_neg_mask[tuple(train_edges_pos.tolist())] = False
            train_edge_neg_mask = torch.triu(train_edge_neg_mask, 1)
            train_edges_neg = torch.nonzero(train_edge_neg_mask).t()

            if (ogbl):

                subgraph_val_edges_pos = subgraph_val.edge_label_index[:, subgraph_val.edge_label.bool()]
                subgraph_test_edges_pos = subgraph_test.edge_label_index[:, subgraph_test.edge_label.bool()]
                
                valid_edge_neg_mask = train_edge_neg_mask.clone()
                valid_edge_neg_mask[tuple(subgraph_val_edges_pos.tolist())] = False
                valid_edges_neg = torch.nonzero(valid_edge_neg_mask).t()

                # Full testing with all negative pairs in the training graph (excluding validation and testing positive edges).
                test_edge_neg_mask = valid_edge_neg_mask.clone()
                test_edge_neg_mask[tuple(subgraph_test_edges_pos.tolist())] = False
                test_edges_neg = torch.nonzero(test_edge_neg_mask).t()

        else:
            train_edges_neg = subgraph_train.edge_label_index[:, ~subgraph_train.edge_label.bool()] 


        splits[idx]['train_edges_pos'] = train_edges_pos
        splits[idx]['train_edges_neg'] = train_edges_neg
        

        
        cluster_mask_val = (intracluster_idx_val == idx)
        cluster_mask_test = (intracluster_idx_test == idx)

        if (ogbl):

            valid_edges_pos = convert_to_partition_index(valid_edges[:, cluster_mask_val], node_to_partition)

            test_edges_pos = convert_to_partition_index(test_edges[:, cluster_mask_test], node_to_partition)

            splits[idx]['valid_edges'] = torch.cat([valid_edges_pos, valid_edges_neg], dim=1)
            splits[idx]['valid_true'] = torch.cat([torch.ones(valid_edges_pos.shape[1]), torch.zeros(valid_edges_neg.shape[1])])

            splits[idx]['test_edges'] = torch.cat([test_edges_pos, test_edges_neg], dim=1)
            splits[idx]['test_true'] = torch.cat([torch.ones(test_edges_pos.shape[1]), torch.zeros(test_edges_neg.shape[1])])

            splits[idx]['valid_edge_index'] = subgraph_val.edge_index
            
            splits[idx]['test_edge_index'] = subgraph_test.edge_index

        else:
            # Selecting the validation edges contained in the cluster 'idx'
            splits[idx]['valid_edges'] = convert_to_partition_index(valid_edges[:, cluster_mask_val], node_to_partition)
            splits[idx]['valid_true'] = valid_true[cluster_mask_val]

            # Selecting the validation edges contained in the cluster 'idx'
            splits[idx]['test_edges'] = convert_to_partition_index(test_edges[:, cluster_mask_test], node_to_partition)
            splits[idx]['test_true'] = test_true[cluster_mask_test]


            cluster_edge_idx_mask_val = (intracluster_edge_idx_val == idx)
            cluster_edge_idx_mask_test = (intracluster_edge_idx_test == idx)

            splits[idx]['valid_edge_index'] = convert_to_partition_index(valid_edge_index[:, cluster_edge_idx_mask_val], node_to_partition)

            splits[idx]['test_edge_index'] = convert_to_partition_index(test_edge_index[:, cluster_edge_idx_mask_test], node_to_partition)


    # Accounting for the intercluster pairs left behind
    intercluster_splits = dict()
    
    intercluster_splits['valid_edges'] = valid_edges[:, intercluster_mask_val]
    intercluster_splits['valid_true'] = valid_true[intercluster_mask_val]

    intercluster_splits['test_edges'] = test_edges[:, intercluster_mask_test]
    intercluster_splits['test_true'] = test_true[intercluster_mask_test]


    torch.save(splits, save_dir + f'splits_{n_partitions}.pt')
    torch.save(intercluster_splits, save_dir + f'intercluster_splits_{n_partitions}.pt')
    torch.save(node_to_partition, save_dir + f'node_to_partition_{n_partitions}.pt')

    return splits, intercluster_splits, node_to_partition


def compute_batch_stats(out, running_sum, running_sum_squared, running_n):
    results = out.detach().clone()
    running_sum += results.sum()
    running_sum_squared += torch.dot(results, results)
    running_n += len(results)
    
    running_mean = running_sum / running_n
    
    running_std = torch.sqrt(running_sum_squared / (running_n - 1)) - (running_mean ** 2)

    # mean = min(out_flatten[valid_idx])
    # std = max(out_flatten[valid_idx]) - min(out_flatten[valid_idx])
    return running_mean, running_std, running_sum, running_sum_squared, running_n


def parse_args():
    """
    Parse the arguments.

    :return: Parsed arguments
    """

    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument('--dataset', default='Cora',
                        help='Dataset. Default is Cora. ')
    parser.add_argument('--eta', type=float, default=0.5,
                        help='Proportion of added edges. Default is 0.0. ')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Topological weight. Default is 0.0. ')
    parser.add_argument('--beta', type=float, default=0.25,
                        help='Trained weight. Default is 1.0. ')
    parser.add_argument('--add-self-loop', type=literal_eval, default=True,
                        help='Whether to add self-loops to all nodes. Default is False. ')
    parser.add_argument('--trained-edge-weight-batch-size', type=int, default=50000,
                        help='Batch size for computing the trained edge weights. Default is 20000. ')
    parser.add_argument('--graph-learning-type', default='mlp',
                        help='Type of the graph learning component. Default is mlp. ')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of layers in mlp. Default is 3. ')
    parser.add_argument('--hidden-channels', type=int, default=128,
                        help='Number of hidden channels in mlp. Default is 128. ')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate. Default is 0.5. ')
    parser.add_argument('--topological-heuristic-type', default='ac',
                        help='Type of the topological heuristic component. Default is ac. ')
    parser.add_argument('--scaling-parameter', type=int, default=3,
                        help='Scaling parameter of ac. Default is 3. ')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate. Default is 0.001. ')
    parser.add_argument('--epochs', type=int, default=250,
                        help='Number of epochs. Default is 250. ')
    parser.add_argument('--runs', type=int, default=3,
                        help='Number of runs. Default is 3. ')
    parser.add_argument('--train-batch-ratio', type=float, default=0.01,
                        help='Ratio of training edges per train batch. Default is 0.01. ')
    parser.add_argument('--val-batch-ratio', type=float, default=0.01,
                        help='Ratio of val edges per train batch. Default is 0.01. ')
    parser.add_argument('--sample-ratio', type=float, default=0.1,
                        help='Ratio of training edges per train batch. Default is 0.01. ')
    parser.add_argument('--random-seed', type=int, default=0,
                        help='Random seed for training. Default is 1. ')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Index of cuda device to use. Default is 0. ')
    parser.add_argument('--num-partitions', type=int, default=1)
    parser.add_argument('--partition-eval', action='store_true')
    parser.add_argument('--sparse-s', type=literal_eval, default=False)
    parser.add_argument('--full-training', action='store_true')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--intracluster-only', action='store_true')
    parser.add_argument('--no-batch', action='store_true')
    parser.add_argument('--heart-path', type=str, default='')
    
    return parser.parse_args()


def load_dataset(dataset):
    """
    Load the dataset from PyG.
    :param dataset: name of the dataset. Options: 'Karate'
    :return: PyG dataset data.
    """
    data_folder = f'data/{dataset}/'
    if dataset in ('Karate'):
        pyg_dataset = KarateClub(data_folder)
    elif dataset in ('Cora', 'CiteSeer', 'PubMed'):
        pyg_dataset = Planetoid(data_folder, dataset)
    elif dataset in ('Photo', 'Computers'):
        pyg_dataset = Amazon(data_folder, dataset)
    elif dataset in ('ogbl-ppa'):
        pyg_dataset = PygLinkPropPredDataset('ogbl-ppa', root=data_folder)
    elif dataset in ('ogbl-ddi', 'ogbl-collab', 'ogbl-ppa', 'ogbl-wikikg2', 'ogbl-vessel', 'ogbl-biokg'):
        pyg_dataset = PygLinkPropPredDataset(dataset, root=data_folder)
    else:
        raise NotImplementedError(f'{dataset} not supported. ')
    data = pyg_dataset.data
    
    return data, pyg_dataset


@torch.jit.script
def cosine_sim(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    return (y @ x.t()) / torch.outer(y.norm(p=2, dim=1), x.norm(p=2, dim=1))



def compute_batches(rows, batch_size, shuffle=True):
    """
    Compute the batches of rows. This implementation is much faster than pytorch's dataloader.
    :param rows: rows to split into batches.
    :param batch_size: size of each batch.
    :param shuffle: whether to shuffle the rows before splitting.
    :return:
    """

    if shuffle:
        return torch.split(rows[torch.randperm(rows.shape[0])], batch_size)
    else:
        return torch.split(rows, batch_size)


def compute_matrices(edges, edges_pos, edge_neighbors, A, training):
    
    # Finding the biggest dfs in the batch.
    # This will be the size of the matrix we'll  consider
    size = 0
    for edge in edges:
        i, j = int(edge[0]), int(edge[1])
        if ((i, j) in edge_neighbors and len(edge_neighbors[(i, j)]) > size):
                size = len(edge_neighbors[(i, j)])


    for idx, edge in enumerate(edges):
        i, j = int(edge[0]), int(edge[1])

        A_tilde = A.index_put(tuple(edges_pos.t()), torch.zeros(
                    edges_pos.shape[0], device=A.device)) if training else A

        neighbors = edge_neighbors[(i, j)]

        # Hash creation step.
        # This will be used to reduce the size of all tensors used in the computation of the autocovatiance
        neighborhood_idx = dict(zip(neighbors, range(0, len(neighbors))))
        edges_idx_converted = [neighborhood_idx[i], neighborhood_idx[j]]

    
    return S, A, X


def matrix_standarization(matrix, size_reduction_indexes, final_dimension):
    
    matrix_reduced = matrix[size_reduction_indexes,:][:,size_reduction_indexes]
    matrix_nonzero_idx = matrix_reduced.nonzero(as_tuple=True)

    final_matrix = torch.zeros((final_dimension, final_dimension))
    final_matrix.index_put_(matrix_nonzero_idx, matrix_reduced[matrix_nonzero_idx].float())

    return final_matrix


def batch_matrix_standarization(matrix, pairs, neighbors, final_dimension):

    number_of_pairs = len(pairs)
    final_matrix = torch.zeros((number_of_pairs, final_dimension, final_dimension))

    for pair_index in range(number_of_pairs):

        current_pair_neighbors = neighbors[pair_index]

        matrix_reduced = matrix[current_pair_neighbors,:][:,current_pair_neighbors]

        matrix_nonzero_idx = matrix_reduced.nonzero(as_tuple=False)
        
        pair_indicator_column = torch.full((len(matrix_nonzero_idx),), pair_index)

        matrix_nonzero_idx = torch.column_stack((pair_indicator_column, torch.tensor(matrix_nonzero_idx)))
        
        final_matrix.index_put_(matrix_nonzero_idx, matrix_reduced[matrix_nonzero_idx].float())

    return final_matrix


def batch_matrix_standarization(matrix, pairs, neighbors, final_dimension):
    """
    Given a batch of matrices of different shape, 
    creates a batch of matrices of the same size.

    :param out_pos: similarity scores for positive pairs.
    :param out_neg: similarity scores for negative pairs.
    :return: loss (normalized by the total number of pairs)
    """

    number_of_pairs = len(pairs)
    final_matrix = torch.zeros((number_of_pairs, final_dimension, final_dimension))

    for pair_index in range(number_of_pairs):

        current_pair_neighbors = neighbors[pair_index]

        matrix_reduced = matrix[current_pair_neighbors,:][:,current_pair_neighbors]

        matrix_nonzero_idx = matrix_reduced.nonzero(as_tuple=False)

        pair_indicator_column = torch.full((len(matrix_nonzero_idx),), pair_index)

        matrix_nonzero_idx = torch.column_stack((pair_indicator_column, matrix_nonzero_idx.clone().detach()))
        # print(matrix_nonzero_idx)
        # code.interact(local=locals())
        final_matrix.index_put_(tuple(matrix_nonzero_idx.t()), matrix_reduced[tuple(matrix_nonzero_idx[:, [1, 2]].t())].float())

    return final_matrix


def sparse_mul_by_scalar(sparse:SparseTensor, scalar:float):
    row, col, value = sparse.coo()
    return sparse.set_value(value * scalar, layout='coo')
