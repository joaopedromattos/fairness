import numpy as np
import scipy.sparse as sp
import torch

def scipysp_to_pytorchsp(sp_mx):
    """ converts scipy sparse matrix to pytorch sparse matrix """
    if not sp.isspmatrix_coo(sp_mx):
        sp_mx = sp_mx.tocoo()
    coords = np.vstack((sp_mx.row, sp_mx.col)).transpose()
    values = sp_mx.data
    shape = sp_mx.shape
    pyt_sp_mx = torch.sparse.FloatTensor(torch.LongTensor(coords.T),
                                         torch.FloatTensor(values),
                                         torch.Size(shape))
    return pyt_sp_mx

def accuracy(output, labels):
    output = output.squeeze()
    preds = (output>0).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def fair_metric(output,idx, labels, sens):
    val_y = labels[idx].cpu().numpy()
    idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()]==0
    idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()]==1

    idx_s0_y1 = np.bitwise_and(idx_s0,val_y==1)
    idx_s1_y1 = np.bitwise_and(idx_s1,val_y==1)

    pred_y = (output[idx].squeeze()>0).type_as(labels).cpu().numpy()
    parity = abs(sum(pred_y[idx_s0])/sum(idx_s0)-sum(pred_y[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred_y[idx_s0_y1])/sum(idx_s0_y1)-sum(pred_y[idx_s1_y1])/sum(idx_s1_y1))

    return parity,equality


def normalize_adjacency(adj, deg):
    # Calculate the degrees
    row, col = adj.indices()
    edge_weight = adj.values() if adj.values() is not None else torch.ones(row.size(0))
    # degree = torch_scatter.scatter_add(edge_weight, row, dim=0, dim_size=adj.size(0))

    # Inverse square root of degree matrix
    degree_inv_sqrt = deg.pow(-0.5)
    degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0

    # Normalize
    row_inv = degree_inv_sqrt[row]
    col_inv = degree_inv_sqrt[col]
    norm_edge_weight = edge_weight * row_inv * col_inv

    # Create the normalized sparse tensor
    adj_norm = torch.sparse.FloatTensor(torch.stack([row, col]), norm_edge_weight, adj.size())
    return adj_norm