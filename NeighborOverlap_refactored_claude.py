import torch
import numpy as np
from torch_geometric.utils import negative_sampling
from torch_geometric.transforms import RandomNodeSplit
from torch.utils.tensorboard import SummaryWriter
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from sklearn.metrics import roc_auc_score, average_precision_score
from torchmetrics.functional.classification import binary_accuracy, accuracy

from model import predictor_dict, convdict, GCN, DropEdge
from utils import PermIterator, fairsample, disable_training, enable_training, is_model_trainable
from fairness import FairLearner, FairLearner_GNN
from node_classifier_module import NodeClassifierModule
from evaluation import (
    true_positive_rate_disparity,
    positive_rate_disparity,
    group_positive_rate_disparity,
    group_true_positive_rate_disparity
)
from args import parseargs

import wandb
import time
import datetime
import os
from pathlib import Path
from loguru import logger

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor

def create_masked_adjacency(data, pos_train_edge, perm):
    """
    Create a masked adjacency matrix for the given permutation.
    """
    adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)
    adjmask[perm] = 0
    tei = pos_train_edge[:, adjmask]
    adj = SparseTensor.from_edge_index(tei,
                        sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(
                            pos_train_edge.device, non_blocking=True)
    return adj.to_symmetric()

def get_predictions(predictor, h, adj, pos_edge, neg_edge, args):
    """
    Get predictions for positive and negative edges.
    """
    if not args.link_level:
        pos_pred = predictor.multidomainforward(h, adj, pos_edge, cndropprobs=args.cndropprobs)
        neg_pred = predictor.multidomainforward(h, adj, neg_edge, cndropprobs=args.cndropprobs)
    else:
        pos_pred, pos_embs = predictor.multidomainforward(h, adj, pos_edge, cndropprobs=args.cndropprobs)
        neg_pred, neg_embs = predictor.multidomainforward(h, adj, neg_edge, cndropprobs=args.cndropprobs)
        
    return pos_pred, neg_pred

def calculate_loss(pos_pred, neg_pred, h, h_flipped, fair_model, data, pos_edge, neg_edge, args):
    """
    Calculate the total loss including the prediction loss and fairness loss.
    """
    pos_loss = -F.logsigmoid(pos_pred).mean()
    neg_loss = -F.logsigmoid(-neg_pred).mean()
    pred_loss = pos_loss + neg_loss

    if args.no_intervention:
        return pred_loss

    if args.nifty:
        if args.link_level:
            z1 = torch.cat([pos_embs, neg_embs])
            z2 = torch.cat([pos_embs_flipped, neg_embs_flipped])
            fairness_loss = NIFTY_loss(z1, z2, args.projection_layer[0], args.projection_layer[1])
        else:
            fairness_loss = NIFTY_loss(h, h_flipped, args.projection_layer[0], args.projection_layer[1])
        
        return pred_loss - (args.reg_lambda * fairness_loss)

    if args.link_level:
        protected_groups_pos = (data.y[pos_edge].sum(0).long() == 1).float()
        protected_groups_neg = (data.y[neg_edge].sum(0).long() == 1).float()
        
        embs = torch.cat([pos_embs, neg_embs])
        labels = torch.cat([protected_groups_pos, protected_groups_neg])
        labels = F.one_hot(labels.long(), num_classes=len(labels.unique())).float()
        
        embs = F.tanh(embs)
        fair_pred = fair_model(embs)
        
        if args.laftr_dp or args.laftr_eo:
            fairness_loss = calculate_laftr_loss(fair_pred, labels, protected_groups_pos, protected_groups_neg, args)
        else:
            fairness_loss = F.binary_cross_entropy_with_logits(fair_pred, labels)
    else:
        h = F.tanh(h)
        if args.fair_learner_gnn:
            row, col, _ = adj.coo()
            edge_index = torch.stack([row, col], dim=0)
            fair_pred = fair_model(h, edge_index)
        else:
            fair_pred = fair_model(h)
        
        labels = F.one_hot(data.y.long(), num_classes=len(data.y.unique())).float()
        
        if args.laftr_eo:
            class_weights = 1 / labels.sum(0)
            fairness_loss = ((fair_pred.softmax(dim=-1) - labels).abs().sum(0) * class_weights).sum()
        else:
            fairness_loss = F.binary_cross_entropy_with_logits(fair_pred.squeeze(-1), labels)

    if args.dp_only:
        return fairness_loss
    else:
        return pred_loss - (args.reg_lambda * fairness_loss.item())

def calculate_laftr_loss(fair_pred, labels, protected_groups_pos, protected_groups_neg, args):
    """
    Calculate the LAFTR (Learning Adversarially Fair and Transferable Representations) loss.
    """
    if args.laftr_dp:
        sens_groups = torch.cat([protected_groups_pos, protected_groups_neg])
        sens_preds = fair_pred.flatten()
        sens_group_proportion = 1 / sens_groups.sum()
        non_sensitive_group_proportion = 1 - sens_group_proportion
        fairness_loss = 1 - (sens_group_proportion * torch.abs(sens_preds[sens_groups.bool()] - sens_groups[sens_groups.bool()])).sum() + \
                            (non_sensitive_group_proportion * torch.abs(sens_preds[~sens_groups.bool()] - sens_groups[~sens_groups.bool()])).sum()
    elif args.laftr_eo:
        pos_pairs_fair_outputs = fair_pred.softmax(dim=-1)[:protected_groups_pos.shape[0], :]
        pos_pairs_fair_labels = labels[:protected_groups_pos.shape[0]]
        class_weights = 1 / pos_pairs_fair_labels.sum(0)
        fairness_loss = ((pos_pairs_fair_outputs - pos_pairs_fair_labels).abs().sum(0) * class_weights).sum()
    else:
        raise ValueError("Invalid LAFTR option")
    
    return fairness_loss

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def update_best_results(results, best_results):
    if best_results is None:
        return results
    return {key: max(result, best_results[key]) for key, result in results.items()}

def NIFTY_loss(z1, z2, projection_layer1, projection_layer2):
    def D(x1, x2):
        return -torch.nn.functional.cosine_similarity(x1, x2.detach(), dim=-1).mean()
    
    p1, p2 = projection_layer1(z1), projection_layer1(z2)
    h1, h2 = projection_layer2(p1), projection_layer2(p2)
    
    return (D(h1, p2) + D(h2, p1)) / 2

def train(model, fair_model, predictor, data, split_edge, optimizer, fair_optimizer, args):
    model.train()
    predictor.train()
    fair_model.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device).t()
    
    total_loss = []
    total_fair_loss = []
    
    for perm in PermIterator(pos_train_edge.device, pos_train_edge.shape[1], args.batch_size):
        optimizer.zero_grad()
        fair_optimizer.zero_grad()
        
        if args.maskinput:
            adj = create_masked_adjacency(data, pos_train_edge, perm)
        else:
            adj = data.adj_t
        
        h = model(data.x, adj)
        
        if args.nifty:
            h_flipped = model(flip_sensitive_feature(data.x), adj)
        
        pos_edge = pos_train_edge[:, perm]
        neg_edge = negative_sampling(pos_edge, num_nodes=data.num_nodes)
        
        pos_pred, neg_pred = get_predictions(predictor, h, adj, pos_edge, neg_edge, args)
        
        loss = calculate_loss(pos_pred, neg_pred, h, h_flipped if args.nifty else None, 
                              fair_model, data, pos_edge, neg_edge, args)
        
        loss.backward()
        optimizer.step()
        fair_optimizer.step()
        
        total_loss.append(loss.item())
        
    return np.mean(total_loss), np.mean(total_fair_loss)

@torch.no_grad()
def test(model, fair_model, predictor, data, split_edge, evaluator, args):
    model.eval()
    predictor.eval()
    fair_model.eval()

    h = model(data.x, data.adj_t)
    
    results = {}
    for split in ['train', 'valid', 'test']:
        pos_edge = split_edge[split]['edge'].to(h.device)
        neg_edge = split_edge[split]['edge_neg'].to(h.device)
        
        pos_pred = get_batch_predictions(predictor, h, data.adj_t, pos_edge, args)
        neg_pred = get_batch_predictions(predictor, h, data.adj_t, neg_edge, args)
        
        results[f'{split}_hits@20'] = evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })['hits@20'].item()
        
        # Calculate fairness metrics
        edge_pred = torch.cat([pos_pred, neg_pred])
        edge_true = torch.cat([torch.ones(pos_edge.size(0)), torch.zeros(neg_edge.size(0))])
        protected_groups = get_protected_groups(data, torch.cat([pos_edge, neg_edge]))
        
        results[f'{split}_tprd'] = true_positive_rate_disparity(edge_true, edge_pred, protected_groups)
        results[f'{split}_prd'] = positive_rate_disparity(edge_pred, protected_groups)
        
    return results, h

def main():
    args = parseargs()
    set_seed(args.seed)
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    
    data, split_edge = load_dataset(args)
    
    model = GCN(data.num_features, args.hidden_dim, args.hidden_dim, args.num_layers, 
                dropout=args.dropout, use_ln=args.use_ln).to(device)
    
    fair_model = (FairLearner_GNN if args.fair_learner_gnn else FairLearner)(
        args.hidden_dim, args.hidden_dim * 2, 4, out_channels=2).to(device)
    
    predictor = get_predictor(args).to(device)
    
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': args.gnn_lr},
        {'params': predictor.parameters(), 'lr': args.pred_lr}
    ])
    
    fair_optimizer = torch.optim.Adam(fair_model.parameters(), lr=args.adv_lr)
    
    evaluator = Evaluator(name='ogbl-ppa')
    
    best_valid_auc = 0
    best_test_auc = 0
    
    for epoch in range(1, args.epochs + 1):
        loss, fair_loss = train(model, fair_model, predictor, data, split_edge, 
                                optimizer, fair_optimizer, args)
        
        results, h = test(model, fair_model, predictor, data, split_edge, evaluator, args)
        
        if results['valid_hits@20'] > best_valid_auc:
            best_valid_auc = results['valid_hits@20']
            best_test_auc = results['test_hits@20']
        
        log_results(epoch, loss, fair_loss, results, best_test_auc)
    
    if args.save_model:
        save_model(model, predictor, args)
    
    if args.node_classification:
        perform_node_classification(h, data, args)

if __name__ == "__main__":
    main()