import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    GINConv,
    SAGEConv,
    DeepGraphInfomax,
    JumpingKnowledge,
)
import os
from loguru import logger
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
from torch.nn.utils import spectral_norm
from torch_geometric.utils import dropout_adj, convert

import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import time
import argparse
import numpy as np
import scipy.sparse as sp
from torch_geometric.typing import SparseTensor

from torch_geometric.data import Data

from torch_geometric.utils import to_undirected
from torch_scatter import scatter_add
from loguru import logger
import concurrent.futures


from NCNC import CN0LinkPredictor, CNLinkPredictor
from SEAL import SEALDataset, GCN_SEAL

# '''DO NOT MODIFY
def find_items_per_group_per_query(data, query_ids, which_query, prot_idx):
    judgments_per_query = find_items_per_query(data, query_ids, which_query)
    prot_idx_per_query = find_items_per_query(prot_idx, query_ids, which_query)
    protected_items_per_query = judgments_per_query[prot_idx_per_query.bool()]
    nonprotected_items_per_query = judgments_per_query[~prot_idx_per_query.bool()]
    return judgments_per_query, protected_items_per_query, nonprotected_items_per_query

def find_items_per_query(data, query_ids, which_query):
    # import code
    # code.interact(local={**locals(), **globals()})
    return data[query_ids == which_query]

def normalized_exposure(group_data, all_data):
    return (torch.sum(topp_prot(group_data, all_data) / torch.log(torch.tensor(2.0)))) / group_data.size(0)

def topp_prot(group_items, all_items):
    return torch.exp(group_items) / torch.sum(torch.exp(all_items))

def exposure_diff(data, query_ids, which_query, prot_idx):
    judgments_per_query, protected_items_per_query, nonprotected_items_per_query = \
        find_items_per_group_per_query(data, query_ids, which_query, prot_idx)

    exposure_prot = normalized_exposure(protected_items_per_query, judgments_per_query)
    exposure_nprot = normalized_exposure(nonprotected_items_per_query, judgments_per_query)
    exposure_diff = torch.max(torch.tensor(0.0), (exposure_nprot - exposure_prot)) ** 2
    
    return exposure_diff

# '''


# def vectorized_exposure_diff(data, query_ids, prot_idx):
#     # Get unique query ids
#     unique_queries = torch.unique(query_ids)
    
#     # Create a mask for each unique query
#     query_masks = query_ids.unsqueeze(1) == unique_queries.unsqueeze(0)
#     query_masks = query_masks.to(data.device)
    
#     # Compute judgments per query
#     judgments_per_query = (data.unsqueeze(1) * query_masks)
    
#     # Compute protected and non-protected masks
#     prot_masks = prot_idx.unsqueeze(1).to(data.device) * query_masks
#     nprot_masks = (~prot_idx.bool()).unsqueeze(1).to(data.device) * query_masks
    
    
#     # Compute protected and non-protected items per query
#     protected_items_per_query = (judgments_per_query * prot_masks)
#     nonprotected_items_per_query = (judgments_per_query * nprot_masks)
    
#     import code
#     code.interact(local={**locals(), **globals()})
#     judgments_per_query = torch.exp(judgments_per_query).sum(dim=0)
#     protected_items_per_query = torch.exp(protected_items_per_query)
#     nonprotected_items_per_query = torch.exp(protected_items_per_query)
    
    
#     # Compute group sizes
#     prot_group_sizes = prot_masks.sum(dim=0)
#     nprot_group_sizes = nprot_masks.sum(dim=0)
    
#     # Compute normalized exposure
#     def normalized_exposure(group_data, all_data, group_sizes):
#         topp = group_data / all_data
#         # import code
#         # code.interact(local={**locals(), **globals()})
#         return (topp / torch.log(torch.tensor(2.0))).sum(dim=0) / group_sizes
    
#     exposure_prot = normalized_exposure(protected_items_per_query, judgments_per_query, prot_group_sizes)
#     exposure_nprot = normalized_exposure(nonprotected_items_per_query, judgments_per_query, nprot_group_sizes)
    
#     # Compute exposure difference
#     exposure_diff = torch.max(torch.tensor(0.0), (exposure_nprot - exposure_prot)) ** 2
    
#     return exposure_diff



# def vectorized_exposure_diff(data, query_ids, prot_idx):
#     # Get unique query ids
#     unique_queries = torch.unique(query_ids)
    
#     # Create a mask for each unique query
#     query_masks = query_ids.unsqueeze(1) == unique_queries.unsqueeze(0)
#     query_masks = query_masks.to(data.device)
    
#     # Compute judgments per query
#     judgments_per_query = (data.unsqueeze(1) * query_masks)
    
#     # Compute protected and non-protected masks
#     prot_masks = prot_idx.unsqueeze(1).to(data.device) * query_masks
#     nprot_masks = (~prot_idx.bool()).unsqueeze(1).to(data.device) * query_masks
    
#     # Compute protected and non-protected items per query
#     protected_items_per_query = (judgments_per_query * prot_masks)
#     nonprotected_items_per_query = (judgments_per_query * nprot_masks)
    
    
#     # We have to re-apply the masks since the exp() operation transforms 0s into 1s
#     judgments_per_query = torch.sum(torch.exp(judgments_per_query) * query_masks)
#     protected_items_per_query = torch.exp(protected_items_per_query) * prot_masks
#     nonprotected_items_per_query = torch.exp(nonprotected_items_per_query) * nprot_masks

    
#     # Compute group sizes
#     prot_group_sizes = prot_masks.sum(dim=0)
#     nprot_group_sizes = nprot_masks.sum(dim=0)
    
#     # Compute normalized exposure
#     def modified_normalized_exposure(group_data, all_data, group_sizes):
#         topp = group_data / all_data
#         return (topp / torch.log(torch.tensor(2.0))).sum(dim=0) / group_sizes
    
#     exposure_prot = modified_normalized_exposure(protected_items_per_query, judgments_per_query, prot_group_sizes)
#     exposure_nprot = modified_normalized_exposure(nonprotected_items_per_query, judgments_per_query, nprot_group_sizes)
    
#     # Compute exposure difference
#     exposure_diff = torch.max(torch.tensor(0.0), (exposure_nprot - exposure_prot)) ** 2
    
#     return exposure_diff


def vectorized_exposure_diff(data, query_ids, prot_idx):
    # Get unique query ids
    unique_queries = torch.unique(query_ids)
    
    # Create a mask for each unique query
    query_masks = query_ids.unsqueeze(1) == unique_queries.unsqueeze(0)
    # query_masks = query_ids.unsqueeze(1) == torch.tensor(which_query).unsqueeze(0)
    query_masks = query_masks.to(data.device)
    
    # Compute judgments per query
    judgments_per_query = (data.unsqueeze(1) * query_masks)
    
    # print("BEFORE", judgments_per_query.sum())
    
    # Compute protected and non-protected masks
    prot_masks = prot_idx.unsqueeze(1).to(data.device) * query_masks
    nprot_masks = (~prot_idx.bool()).unsqueeze(1).to(data.device) * query_masks
    
    
    # Compute protected and non-protected items per query
    protected_items_per_query = (judgments_per_query * prot_masks)
    nonprotected_items_per_query = (judgments_per_query * nprot_masks)
    
    # print("Before exp", nprot_masks)
    
    import code
    code.interact(local={**locals(), **globals()})

    # We have to re-apply the masks since the exp() operation transforms 0s into 1s
    judgments_per_query = torch.sum(torch.exp(judgments_per_query) * query_masks)
    protected_items_per_query = torch.exp(protected_items_per_query) * prot_masks
    nonprotected_items_per_query = torch.exp(nonprotected_items_per_query) * nprot_masks
    # return judgments_per_query, protected_items_per_query, nonprotected_items_per_query
    
    # print("AFTER", judgments_per_query.type())
    # print("Non zeros -----")
    # print(judgments_per_query, protected_items_per_query.squeeze()[(judgments_per_query * prot_masks).squeeze().nonzero()], nonprotected_items_per_query.squeeze()[(judgments_per_query * nprot_masks).squeeze().nonzero()])
    # print(judgments_per_query.shape, protected_items_per_query.shape, nonprotected_items_per_query.shape)
    
    
    # Compute group sizes
    prot_group_sizes = prot_masks.sum(dim=0)
    nprot_group_sizes = nprot_masks.sum(dim=0)
    
    # Compute normalized exposure
    def modified_normalized_exposure(group_data, all_data, group_sizes):
        topp = group_data / all_data
        # import code
        # code.interact(local={**locals(), **globals()})
        return (topp / torch.log(torch.tensor(2.0))).sum(dim=0) / group_sizes
    
    exposure_prot = modified_normalized_exposure(protected_items_per_query, judgments_per_query, prot_group_sizes)
    exposure_nprot = modified_normalized_exposure(nonprotected_items_per_query, judgments_per_query, nprot_group_sizes)
    
    # import code
    # code.interact(local={**locals(), **globals()})
    
    # Compute exposure difference
    exposure_difference = torch.max(torch.tensor(0.0), (exposure_nprot - exposure_prot)) ** 2
    
    return exposure_difference




# @torch.compile
def n_pair_loss(out_pos, out_neg):
    """
    Compute the N-pair loss.

    :param out_pos: similarity scores for positive pairs.
    :param out_neg: similarity scores for negative pairs.
    :return: loss (normalized by the total number of pairs)
    """

    agg_size = out_neg.shape[0] // out_pos.shape[0]  # Number of negative pairs matched to a positive pair.
    agg_size_p1 = agg_size + 1
    agg_size_p1_count = out_neg.shape[0] % out_pos.shape[0]  # Number of positive pairs that should be matched to agg_size + 1 instead because of the remainder.
    out_pos_agg_p1 = out_pos[:agg_size_p1_count].unsqueeze(-1)
    out_pos_agg = out_pos[agg_size_p1_count:].unsqueeze(-1)
    out_neg_agg_p1 = out_neg[:agg_size_p1_count * agg_size_p1].reshape(-1, agg_size_p1)
    out_neg_agg = out_neg[agg_size_p1_count * agg_size_p1:].reshape(-1, agg_size)
    out_diff_agg_p1 = out_neg_agg_p1 - out_pos_agg_p1  # Difference between negative and positive scores.
    out_diff_agg = out_neg_agg - out_pos_agg  # Difference between negative and positive scores.
    out_diff_exp_sum_p1 = torch.exp(torch.clamp(out_diff_agg_p1, max=80.0)).sum(axis=1)
    out_diff_exp_sum = torch.exp(torch.clamp(out_diff_agg, max=80.0)).sum(axis=1)
    out_diff_exp_cat = torch.cat([out_diff_exp_sum_p1, out_diff_exp_sum])
    loss = torch.log(1 + out_diff_exp_cat).sum() / (len(out_pos) + len(out_neg))

    return loss


# @torch.compile
def vectorized_edge_level_exposure_diff(data, prot_idx):
    
    data = (data - data.mean()) / data.std()

    data = torch.clamp(data, max=80.0)
    
    prot_scores = data[prot_idx.bool()]
    nprot_scores = data[(~prot_idx).bool()]
    
    
    # top-1 probs
    topp_prot = torch.exp(prot_scores) / torch.sum(torch.exp(data))
    topp_nprot = torch.exp(nprot_scores) / torch.sum(torch.exp(data))
    
    prot_size = prot_idx.sum()
    nprot_size = (~prot_idx).sum()
    
    exposure_prot = (topp_prot / torch.log(torch.tensor(2.0))).sum() / prot_size
    exposure_nprot = (topp_nprot / torch.log(torch.tensor(2.0))).sum() / nprot_size
    
    exposure_difference = torch.max(torch.tensor(0.0), exposure_nprot - exposure_prot) ** 2 
    
    
    
    # import code
    # code.interact(local={**locals(), **globals()})
    
    return exposure_difference




# def find_items_per_group_per_query(data, query_ids, prot_idx):
#     prot_idx_per_query = prot_idx.bool()
#     protected_items_per_query = data[prot_idx_per_query]
#     nonprotected_items_per_query = data[~prot_idx_per_query]
#     return data, protected_items_per_query, nonprotected_items_per_query

# # No need for a specific query selection since we handle all in one go
# def find_items_per_query(data, query_ids):
#     return data


# def exposure_diff_batched(data, query_ids, prot_idx):
#     # Assume data, query_ids, and prot_idx are all 1D tensors

#     # Find items per group per query for the whole batch
#     judgments_per_query, protected_items_per_query, nonprotected_items_per_query = \
#         find_items_per_group_per_query(data, query_ids, prot_idx)
    
#     # Calculate normalized exposure for protected and non-protected groups
#     exposure_prot = normalized_exposure(protected_items_per_query, judgments_per_query)
#     exposure_nprot = normalized_exposure(nonprotected_items_per_query, judgments_per_query)
    
#     # Calculate exposure difference for each query
#     exposure_diff = torch.max(torch.tensor(0.0), exposure_nprot - exposure_prot)

#     # Now scatter the results based on the query_ids
#     exposure_diff_scatter = scatter_add(exposure_diff, query_ids)

#     return exposure_diff_scatter.sum()



'''MY ATTEMPT USING TORCH SCATTER
# judgments_per_query = scatter_sum(torch.ones(len(query_ids)), query_ids)
# prot_items_per_query = scatter_sum(prot_idx.float(), query_ids)
# non_prot_items_per_query = scatter_sum((torch.ones_like(prot_idx) - prot_idx).float(), query_ids)
'''


# def find_items_per_group(data, query_ids, prot_idx):
#     # Create a mask for each unique query
#     unique_queries = torch.unique(query_ids)
#     import code
#     code.interact(local={**locals(), **globals()})
#     query_mask = query_ids.unsqueeze(0) == unique_queries.unsqueeze(1)
    
#     # Use broadcasting to select data for each query
#     judgments_per_query = data.unsqueeze(0) * query_mask.unsqueeze(-1)
#     prot_mask = prot_idx.unsqueeze(0) * query_mask
    
#     protected_items = judgments_per_query * prot_mask.unsqueeze(-1)
#     nonprotected_items = judgments_per_query * (~prot_mask).unsqueeze(-1)
    
#     return judgments_per_query, protected_items, nonprotected_items


# def normalized_exposure(group_data, all_data):
#     exp_group = torch.exp(group_data)
#     exp_all = torch.exp(all_data)
    
#     topp_prot = exp_group / (exp_all.sum(dim=1, keepdim=True) + 1e-10)
#     exposure = (topp_prot / torch.log(torch.tensor(2.0))).sum(dim=1)
    
#     group_size = (group_data != 0).sum(dim=1)
#     return exposure / (group_size + 1e-10)


# def exposure_diff(data, query_ids, prot_idx):
    
#     query_ids = to_undirected(query_ids)
    
#     judgments_per_query, protected_items, nonprotected_items = find_items_per_group(data, query_ids, prot_idx)
    
#     exposure_prot = normalized_exposure(protected_items, judgments_per_query)
#     exposure_nprot = normalized_exposure(nonprotected_items, judgments_per_query)
    
#     exposure_diff = torch.max(torch.tensor(0.0), (exposure_nprot - exposure_prot))
    
#     return exposure_diff



class Classifier(nn.Module):
    def __init__(self, ft_in, nb_classes, base_model="standard"):
        super(Classifier, self).__init__()
        
        self.base_model = base_model

        # Classifier projector
        if base_model == 'classifier':
            self.predictor = spectral_norm(nn.Linear(ft_in, nb_classes))
        elif base_model == 'gae':
            self.predictor = CN0LinkPredictor(in_channels=ft_in, hidden_channels=ft_in, out_channels=nb_classes, num_layers=2, dropout=0.5)
        elif base_model == 'ncn':
            self.predictor = CNLinkPredictor(in_channels=ft_in, hidden_channels=ft_in, out_channels=nb_classes, num_layers=2, dropout=0.5)
        elif base_model == 'seal':
            self.predictor = nn.Linear(ft_in, 2)

    def forward(self, x, adj, tar_ei):
        ret = None
        if self.base_model == 'standard':
            ret = self.predictor(x)
        elif self.base_model == 'gae':
            ret = self.predictor.multidomainforward(x, adj, tar_ei)
        elif self.base_model == 'ncn':
            adj = SparseTensor.from_edge_index(adj, sparse_sizes=(x.size(0), x.size(0)))
            ret = self.predictor.multidomainforward(x, adj, tar_ei)
        elif self.base_model == 'seal':
            ret = self.predictor(x)
            
        return ret


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        return x


class GIN(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(GIN, self).__init__()

        self.mlp1 = nn.Sequential(
            spectral_norm(nn.Linear(nfeat, nhid)),
            nn.ReLU(),
            nn.BatchNorm1d(nhid),
            spectral_norm(nn.Linear(nhid, nhid)),
        )
        self.conv1 = GINConv(self.mlp1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x


class JK(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(JK, self).__init__()
        self.conv1 = spectral_norm(GCNConv(nfeat, nhid))
        self.convx = spectral_norm(GCNConv(nhid, nhid))
        self.jk = JumpingKnowledge(mode="max")
        self.transition = nn.Sequential(
            nn.ReLU(),
        )

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        xs = []
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        xs.append(x)
        for _ in range(1):
            x = self.convx(x, edge_index)
            x = self.transition(x)
            xs.append(x)
        x = self.jk(xs)
        return x


class SAGE(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(SAGE, self).__init__()

        # Implemented spectral_norm in the sage main file
        # ~/anaconda3/envs/PYTORCH/lib/python3.7/site-packages/torch_geometric/nn/conv/sage_conv.py
        self.conv1 = SAGEConv(nfeat, nhid, normalize=True)
        self.conv1.aggr = "mean"
        self.transition = nn.Sequential(
            nn.ReLU(), nn.BatchNorm1d(nhid), nn.Dropout(p=dropout)
        )
        self.conv2 = SAGEConv(nhid, nhid, normalize=True)
        self.conv2.aggr = "mean"

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        x = self.conv2(x, edge_index)
        return x


class Encoder_DGI(nn.Module):
    def __init__(self, nfeat, nhid):
        super(Encoder_DGI, self).__init__()
        self.hidden_ch = nhid
        self.conv = spectral_norm(GCNConv(nfeat, self.hidden_ch))
        self.activation = nn.PReLU()

    def corruption(self, x, edge_index):
        # corrupted features are obtained by row-wise shuffling of the original features
        # corrupted graph consists of the same nodes but located in different places
        return x[torch.randperm(x.size(0))], edge_index

    def summary(self, z, *args, **kwargs):
        return torch.sigmoid(z.mean(dim=0))

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.activation(x)
        return x


class GraphInfoMax(nn.Module):
    def __init__(self, enc_dgi):
        super(GraphInfoMax, self).__init__()
        self.dgi_model = DeepGraphInfomax(
            enc_dgi.hidden_ch, enc_dgi, enc_dgi.summary, enc_dgi.corruption
        )

    def forward(self, x, edge_index):
        pos_z, neg_z, summary = self.dgi_model(x, edge_index)
        return pos_z


class Encoder(torch.nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, max_z:int, base_model="gcn", k: int = 2
    ):
        super(Encoder, self).__init__()
        self.base_model = base_model
        if self.base_model == "gcn":
            self.conv = GCN(in_channels, out_channels)
        elif self.base_model == "gin":
            self.conv = GIN(in_channels, out_channels)
        elif self.base_model == "sage":
            self.conv = SAGE(in_channels, out_channels)
        elif self.base_model == "infomax":
            enc_dgi = Encoder_DGI(nfeat=in_channels, nhid=out_channels)
            self.conv = GraphInfoMax(enc_dgi=enc_dgi)
        elif self.base_model == "jk":
            self.conv = JK(in_channels, out_channels)
        elif self.base_model == "seal":
            self.conv = GCN_SEAL(num_features=in_channels, hidden_channels=out_channels, num_layers=3, max_z=max_z)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, z=None, batch=None, edge_weight=None, node_id=None) -> torch.Tensor:
        if self.base_model == "seal":
            print("Inside encoder")
            import code
            code.interact(local={**locals(), **globals()})
            return self.conv(z, edge_index, batch, x, edge_weight, node_id)
        else:
            return self.conv(x, edge_index)
        


class DELTR(torch.nn.Module):
    def __init__(
        self,
        adj,
        features,
        labels,
        idx_train,
        idx_val,
        idx_test,
        sens,
        sens_idx,
        edge_splits,
        dataset_name,
        num_hidden=16,
        num_proj_hidden=16,
        lr=0.0001,
        weight_decay=1e-5,
        drop_edge_rate_1=0.1,
        drop_feature_rate_1=0.1,
        encoder="gcn",
        decoder='standard',
        exposure_coeff=0.5,
        nclass=1,
        device="cuda",
    ):
        super(DELTR, self).__init__()

        self.device = device

        # self.edge_index = convert.from_scipy_sparse_matrix(sp.coo_matrix(adj.to_dense().numpy()))[0]
        self.edge_index = adj.coalesce().indices()
        self.decoder_name = decoder
        self.encoder_name = encoder
        self.dataset_name = dataset_name
        self.encoder = Encoder(
            in_channels=features.shape[1], out_channels=num_hidden, base_model=encoder, max_z=features.shape[0]
        ).to(device)
        # model = SSF(encoder=encoder, num_hidden=args.hidden, num_proj_hidden=args.proj_hidden, sim_coeff=args.sim_coeff,
        # nclass=num_class).to(device)
        self.val_edge_index_1 = dropout_adj(
            self.edge_index.to(device), p=drop_edge_rate_1
        )[0]

        self.val_x_1 = drop_feature(
            features.to(device), drop_feature_rate_1, sens_idx, sens_flag=False
        )
        
        self.exposure_coeff = exposure_coeff
        # self.encoder = encoder
        self.labels = labels

        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.sens = sens
        self.sens_idx = sens_idx
        self.drop_edge_rate = 0
        self.drop_feature_rate = 0

        # Classifier
        self.c1 = Classifier(ft_in=num_hidden, nb_classes=nclass, base_model=decoder)

        for m in self.modules():
            self.weights_init(m)

        params = list(self.c1.parameters()) + list(self.encoder.parameters())
        self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        
        self = self.to(device)

        self.features = features.to(device)
        self.edge_index = self.edge_index.to(device)
        self.labels = self.labels.to(device)
        
        # Adds all edges and labels as attributes
        for edge_set in edge_splits.keys():
            
            if edge_set != 'test':
                edge_splits[edge_set]['edge'] = to_undirected(edge_splits[edge_set]['edge'].t())
                edge_splits[edge_set]['edge_neg'] = to_undirected(edge_splits[edge_set]['edge_neg'].t())
            else:
                edge_splits[edge_set]['edge'] = edge_splits[edge_set]['edge'].t()
                edge_splits[edge_set]['edge_neg'] = edge_splits[edge_set]['edge_neg'].t()
                
            setattr(self, f'{edge_set}_edge_index', torch.cat([edge_splits[edge_set]['edge'], edge_splits[edge_set]['edge_neg']], dim=-1))
            setattr(self, f'{edge_set}_edge_labels', torch.cat([torch.ones(edge_splits[edge_set]['edge'].size(1)), 
                                                                          torch.zeros(edge_splits[edge_set]['edge_neg'].size(1))]).float().to(self.device))
            
            # We consider sensitive the edges between protected and non protected nodes
            setattr(self, f'{edge_set}_edge_sens', self.sens[getattr(self, f'{edge_set}_edge_index')].sum(0) == 1)
            
        if encoder == 'seal' and decoder == 'seal':
            data = Data(x=self.features, edge_index=self.edge_index, y=self.labels, train_mask=self.idx_train, val_mask=self.idx_val, test_mask=self.idx_test)
            self.seal_dataset = SEALDataset(root='data', data=data.cpu(), split_edge=edge_splits, num_hops=2, split='train').to(device)
            

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, z=None, batch=None, edge_weight=None, node_id=None) -> torch.Tensor:
        return self.encoder(x, edge_index, z, batch, edge_weight, node_id)

    def classifier(self, x, adj=None, tar_ei=None):
        return self.c1(x, adj, tar_ei)

    def normalize(self, x):
        val = torch.norm(x, p=2, dim=1).detach()
        x = x.div(val.unsqueeze(dim=1).expand_as(x))
        return x

    def forwarding_predict(self, emb, adj=None, edge_index=None):

        # classifier
        c1 = self.classifier(emb, adj, edge_index)

        return c1

    def linear_eval(self, emb, labels, idx_train, idx_test):
        x = emb.detach()
        classifier = nn.Linear(in_features=x.shape[1], out_features=2, bias=True)
        classifier = classifier.to("cuda")
        optimizer = torch.optim.Adam(
            classifier.parameters(), lr=0.001, weight_decay=1e-4
        )
        for i in range(1000):
            optimizer.zero_grad()
            preds = classifier(x[idx_train])
            loss = F.cross_entropy(preds, labels[idx_train])
            loss.backward()
            optimizer.step()
            # if i%100==0:
            #     print(loss.item())
        classifier.eval()
        preds = classifier(x[idx_test]).argmax(dim=1)
        correct = (preds == labels[idx_test]).sum().item()
        return preds, correct / preds.shape[0]

    def ssf_validation(self, x_1, edge_index_1, y):
        z1 = self.forward(x_1, edge_index_1)
        
        # classifier
        c1 = self.classifier(z1, edge_index_1, self.valid_edge_index)
        # c1 = (c1 - c1.mean()) / c1.std()
        # exposure_loss = self.exposure_coeff * torch.sum([exposure_diff(c1.squeeze(), self.valid_edge_index[0], query, self.valid_edge_sens) for query in tqdm(self.valid_edge_index[0])])
        # exposure_loss = self.exposure_coeff * torch.sum(vectorized_exposure_diff(c1.squeeze(), self.valid_edge_index[0], self.valid_edge_sens))
        exposure_loss = self.exposure_coeff * vectorized_edge_level_exposure_diff(c1.squeeze(), self.valid_edge_sens)
        # import code
        # code.interact(local={**locals(), **globals()})
        
        # Binary Cross-Entropy
        bce_loss = (F.binary_cross_entropy(c1.sigmoid().squeeze(),self.valid_edge_labels,) / 2)
        bce_loss = bce_loss / self.valid_edge_index.shape[-1] # average per validation sample
        # ranks = torch.softmax(c1.squeeze(), -1)
        # bce_loss = n_pair_loss(ranks[self.valid_edge_sens.bool()], ranks[(~self.valid_edge_sens).bool()])

        return exposure_loss, bce_loss

    def fit(self, epochs=300):

        # Train model
        t_total = time.time()
        best_loss = torch.inf
        best_acc = 0
        best_epoch = 0
        
        for epoch in tqdm(range(epochs + 1)):
            t = time.time()

            exposure_loss = 0
            cl_loss = 0
            rep = range(1, 2) if self.encoder_name != "seal" else self.seal_dataset
            for _ in rep:
                self.train()
                self.optimizer.zero_grad()
                edge_index = dropout_adj(self.edge_index, p=self.drop_edge_rate)[0]
                x = drop_feature(
                    self.features,
                    self.drop_feature_rate,
                    self.sens_idx,
                    sens_flag=False,
                )
                
                if self.encoder_name == 'seal':
                    z1 = self.forward(x, edge_index, z=_.z, batch=_.batch, edge_weight=_.edge_weight, node_id=_.node_id)
                else:
                    z1 = self.forward(x, edge_index)

            
                # classifier
                z1 = self.forward(x, edge_index)
                c1 = self.classifier(z1, edge_index, self.train_edge_index)
                c1 = c1.squeeze()
                # c1 = (c1 - c1.mean()) / c1.std()
                
                # import code
                # code.interact(local={**locals(), **globals()})
                
                # with concurrent.futures.ProcessPoolExecutor() as executor:
                #     futures = [executor.submit(exposure_diff, c1.squeeze(), self.valid_edge_index[0], query, self.valid_edge_sens) for query in tqdm(self.valid_edge_index[0])]
                #     results = [future.result() for future in concurrent.futures.as_completed(futures)]
                # exposure_loss = self.exposure_coeff * torch.sum(results)
                
                
                # exposure_loss += self.exposure_coeff * torch.sum(torch.tensor([exposure_diff(c1.squeeze(), self.train_edge_index[0], query, self.train_edge_sens) for query in tqdm(self.train_edge_index[0])]))
                # print(exposure_loss)
                # exposure_loss += self.exposure_coeff * torch.sum(vectorized_exposure_diff(c1.squeeze(), self.train_edge_index[0], self.train_edge_sens))
                exposure_loss += self.exposure_coeff * vectorized_edge_level_exposure_diff(c1, self.train_edge_sens)
                
                exposure_loss = (exposure_loss / _)

                # Binary Cross-Entropy
                bce_loss = (F.binary_cross_entropy(c1.sigmoid().squeeze(),self.train_edge_labels,) / 2)
                bce_loss = bce_loss / self.train_edge_index.shape[-1] # average per train sample
                loss = bce_loss + exposure_loss
                
                # ranks = c1
                # bce_loss = n_pair_loss(ranks[self.train_edge_sens.bool()], ranks[~self.train_edge_sens.bool()])                
                # import code
                # code.interact(local={**locals(), **globals()})
                loss.backward()
                self.optimizer.step()
                loss = loss.item()
                cl_loss += bce_loss.item()

            # Validation
            self.eval()
            val_s_loss, val_c_loss = self.ssf_validation(
                self.val_x_1,
                self.val_edge_index_1,
                self.labels,
            )
            emb = self.forward(self.val_x_1, self.val_edge_index_1)
            output = self.forwarding_predict(emb, self.val_edge_index_1, self.valid_edge_index)
            preds = (output.squeeze() > 0).type_as(self.labels)
            # auc_roc_val = roc_auc_score(
            #     self.valid_edge_labels.detach().cpu().numpy(),
            #     output.detach().cpu().numpy(),
            # )

            if epoch % 50 == 0:
                logger.info(f"[Train] Epoch {epoch}:train_s_loss: {(exposure_loss/_):.8f} | train_c_loss: {bce_loss:.4f} | val_s_loss: {val_s_loss:.8f} | val_c_loss: {val_c_loss:.4f}")

            if (val_c_loss + val_s_loss) < best_loss:
                best_epoch = epoch
                self.val_loss = val_c_loss.item() + val_s_loss.item()

                logger.info(f'NEW BEST EPOCH - {epoch} | {val_s_loss:.8f} | {val_c_loss:.8f}')
                best_loss = val_c_loss + val_s_loss
                if not os.path.exists("data"):
                    os.makedirs("data")
                torch.save(self.state_dict(), f"data/DELTR_weights_ssf_{self.dataset_name}_{self.encoder_name}_{self.decoder_name}.pt")
                
        logger.info(f"Optimization finished - BEST EPOCH {best_epoch}")
            

    def predict_GNN(self):

        self.load_state_dict(torch.load(f"data/DELTR_weights_ssf_{self.dataset_name}_{self.encoder_name}_{self.decoder_name}.pt"))
        self.eval()
        emb = self.forward(
            self.features.to(self.device), self.edge_index.to(self.device)
        )
        output = self.forwarding_predict(emb)

        output_preds = (
            (output.squeeze() > 0)
            .type_as(self.labels)[self.idx_test]
            .detach()
            .cpu()
            .numpy()
        )

        labels = self.labels.detach().cpu().numpy()
        idx_test = self.idx_test

        F1 = f1_score(labels[idx_test], output_preds, average="micro")
        ACC = accuracy_score(
            labels[idx_test],
            output_preds,
        )
        try:
            AUCROC = roc_auc_score(labels[idx_test], output_preds)
        except:
            AUCROC = "N/A"

        ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1 = (
            self.predict_sens_group(output_preds, idx_test)
        )

        SP, EO = self.fair_metric(
            output_preds,
            self.labels[idx_test].detach().cpu().numpy(),
            self.sens[idx_test].detach().cpu().numpy(),
        )

        return (
            ACC,
            AUCROC,
            F1,
            ACC_sens0,
            AUCROC_sens0,
            F1_sens0,
            ACC_sens1,
            AUCROC_sens1,
            F1_sens1,
            SP,
            EO,
        )

    def predict(self):
        global evaluating
        evaluating = True

        self.load_state_dict(torch.load(f"data/DELTR_weights_ssf_{self.dataset_name}_{self.encoder_name}_{self.decoder_name}.pt"))
        self.eval()
        emb = self.forward(
            self.features.to(self.device), self.edge_index.to(self.device)
        )
        output = self.forwarding_predict(emb, self.edge_index, self.test_edge_index).sigmoid()
        
        return output


    def fair_metric(self, pred, labels, sens):

        idx_s0 = sens == 0
        idx_s1 = sens == 1
        idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
        idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
        parity = abs(sum(pred[idx_s0]) / sum(idx_s0) - sum(pred[idx_s1]) / sum(idx_s1))
        equality = abs(
            sum(pred[idx_s0_y1]) / sum(idx_s0_y1)
            - sum(pred[idx_s1_y1]) / sum(idx_s1_y1)
        )
        return parity.item(), equality.item()

    def predict_sens_group(self, output, idx_test):
        # pred = self.lgreg.predict(self.embs[idx_test])
        pred = output
        result = []
        for sens in [0, 1]:
            F1 = f1_score(
                self.labels[idx_test][self.sens[idx_test] == sens]
                .detach()
                .cpu()
                .numpy(),
                pred[self.sens[idx_test] == sens],
                average="micro",
            )
            ACC = accuracy_score(
                self.labels[idx_test][self.sens[idx_test] == sens]
                .detach()
                .cpu()
                .numpy(),
                pred[self.sens[idx_test] == sens],
            )
            try:
                AUCROC = roc_auc_score(
                    self.labels[idx_test][self.sens[idx_test] == sens]
                    .detach()
                    .cpu()
                    .numpy(),
                    pred[self.sens[idx_test] == sens],
                )
            except:
                AUCROC = "N/A"
            result.extend([ACC, AUCROC, F1])

        return result


def drop_feature(x, drop_prob, sens_idx, sens_flag=True):
    drop_mask = (
        torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1)
        < drop_prob
    )

    x = x.clone()
    drop_mask[sens_idx] = False

    x[:, drop_mask] += torch.ones(1).normal_(0, 1).to(x.device)

    # Flip sensitive attribute
    if sens_flag:
        x[:, sens_idx] = 1 - x[:, sens_idx]

    return x
