import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from ogb.linkproppred import PygLinkPropPredDataset
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges, negative_sampling, to_undirected
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data, Dataset
import os
from tqdm import tqdm

# random split dataset
def randomsplit(dataset, val_ratio: float=0.10, test_ratio: float=0.2):
    def removerepeated(ei):
        ei = to_undirected(ei)
        ei = ei[:, ei[0]<ei[1]]
        return ei
    
    if isinstance(dataset, Dataset):
        data = dataset[0]
    else:
        data = dataset
    
    data.num_nodes = data.x.shape[0]
    data = train_test_split_edges(data, test_ratio, test_ratio)
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    num_val = int(data.val_pos_edge_index.shape[1] * val_ratio/test_ratio)
    data.val_pos_edge_index = data.val_pos_edge_index[:, torch.randperm(data.val_pos_edge_index.shape[1])]
    split_edge['train']['edge'] = removerepeated(torch.cat((data.train_pos_edge_index, data.val_pos_edge_index[:, :-num_val]), dim=-1)).t()
    split_edge['valid']['edge'] = removerepeated(data.val_pos_edge_index[:, -num_val:]).t()
    split_edge['valid']['edge_neg'] = removerepeated(data.val_neg_edge_index).t()
    split_edge['test']['edge'] = removerepeated(data.test_pos_edge_index).t()
    split_edge['test']['edge_neg'] = removerepeated(data.test_neg_edge_index).t()
    return split_edge


def mask_graphair(split_edge, edge_index):
    mask = torch.zeros(edge_index.size(1), dtype=torch.bool).cuda()
    for edge in tqdm(split_edge):
        mask |= ((edge[0] == edge_index).any(0) & (edge[1] == edge_index).any(0))           
    return mask


def loaddataset(name: str, dataset_path:str, use_valedges_as_input: bool, load=None, args=None):
    
    if '_graphair' in args.dataset:
        new_dataset_name = args.dataset.replace("_graphair", "").replace("_contrastive", "")
        
        # We load the current dataset and transform it into the format used by NCN.
        data = torch.load(dataset_path)
        data.edge_index = to_undirected(data.edge_index)
        edge_index = data.edge_index
        data.num_nodes = data.x.shape[0]
            
        data.edge_weight = None 
        print(data.num_nodes, edge_index.max())
        data.adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
        data.adj_t = data.adj_t.to_symmetric().coalesce()
        data.max_x = -1
        
        # Then we load the same edge splits used by all our link prediction experiments.
        dataset_file = f"dataset/splits/{new_dataset_name}{'_node_split' if args.node_split else ''}.pt"
        if os.path.isfile(dataset_file):
            _, split_edge = torch.load(dataset_file)

        # Finally, we mask the test and validation edges in the augmented graph
        # so that we can compare the performance of the models in the same setting.
        mask = mask_graphair(torch.cat([split_edge['train']['edge_neg'], split_edge['valid']['edge_neg'], split_edge['test']['edge_neg'], split_edge['valid']['edge'], split_edge['test']['edge']]).cuda(), data.edge_index.cuda())
        split_edge['train']['edge'] = data.edge_index.t()[mask, :]
        
        # torch.save((data, split_edge), f"dataset/splits/{args.dataset}.pt")
                
    else:
        if name in ["Cora", "Citeseer", "Pubmed"]:
            dataset = Planetoid(root="dataset", name=name)
            split_edge = randomsplit(dataset)
            data = dataset[0]
            data.edge_index = to_undirected(split_edge["train"]["edge"].t())
            edge_index = data.edge_index
            data.num_nodes = data.x.shape[0]

        data = torch.load(dataset_path)
        split_edge = randomsplit(data)
        data.edge_index = to_undirected(split_edge["train"]["edge"].t())
        edge_index = data.edge_index
        data.num_nodes = data.x.shape[0]
            
        data.edge_weight = None 
        print(data.num_nodes, edge_index.max())
        data.adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
        data.adj_t = data.adj_t.to_symmetric().coalesce()
        data.max_x = -1
        if name == "ppa":
            data.x = torch.argmax(data.x, dim=-1)
            data.max_x = torch.max(data.x).item()
        elif name == "ddi":
            data.x = torch.arange(data.num_nodes)
            data.max_x = data.num_nodes
        if load is not None:
            data.x = torch.load(load, map_location="cpu")
            data.max_x = -1
            
        split_edge['train']['edge_neg'] = negative_sampling(data.train_pos_edge_index, data.adj_t.sizes()[0]).t()
        
        if args.node_split:
            mask = torch.zeros(data.num_nodes).bool()
            mask_test = torch.zeros(data.num_nodes).bool()
            idx = torch.randperm(data.num_nodes)
            train_val_idx = idx[:int(idx.shape[0] * 0.8)]
            test_idx = idx[int(idx.shape[0] * 0.8):]
            mask[train_val_idx] = True
            mask_test[test_idx] = True
            
            split_edge['train']['edge'] = split_edge['train']['edge'][mask[split_edge['train']['edge']].all(1)]
            split_edge['train']['edge_neg'] = split_edge['train']['edge_neg'][mask[split_edge['train']['edge_neg']].all(1)]
            split_edge['valid']['edge'] = split_edge['valid']['edge'][mask[split_edge['valid']['edge']].all(1)]
            split_edge['valid']['edge_neg'] = split_edge['valid']['edge_neg'][mask[split_edge['valid']['edge_neg']].all(1)]
            split_edge['test']['edge'] = split_edge['test']['edge'][mask_test[split_edge['test']['edge']].all(1)]
            split_edge['test']['edge_neg'] = split_edge['test']['edge_neg'][mask_test[split_edge['test']['edge_neg']].all(1)]

        print("dataset split ")
        for key1 in split_edge:
            for key2  in split_edge[key1]:
                print(key1, key2, split_edge[key1][key2].shape[0])


        # Use training + validation edges for inference on test set.
        if use_valedges_as_input:
            val_edge_index = split_edge['valid']['edge'].t()
            full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
            data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
            data.full_adj_t = data.full_adj_t.to_symmetric()
        else:
            data.full_adj_t = data.adj_t
    return data, split_edge

if __name__ == "__main__":
    loaddataset("Cora", False)
    loaddataset("Citeseer", False)
    loaddataset("Pubmed", False)
    loaddataset("ppa", False)
    loaddataset("collab", False)
    loaddataset("citation2", False)