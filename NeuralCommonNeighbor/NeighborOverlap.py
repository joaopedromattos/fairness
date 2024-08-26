import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from model import predictor_dict, convdict, GCN, DropEdge
from functools import partial
from sklearn.metrics import roc_auc_score, average_precision_score
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.utils import negative_sampling
from torch_geometric.transforms import RandomNodeSplit
from torch.utils.tensorboard import SummaryWriter
from utils import PermIterator, fairsample, disable_training, enable_training, is_model_trainable
import time
from ogbdataset import loaddataset
from typing import Iterable
from fairness import FairLearner, FairLearner_GNN
from node_classifier_module import NodeClassifierModule
import wandb
from evaluation import true_positive_rate_disparity, positive_rate_disparity, group_positive_rate_disparity, group_true_positive_rate_disparity
import datetime
import os
from loguru import logger
from torchmetrics.functional.classification import binary_accuracy, accuracy
from pathlib import Path
from args import parseargs

torch.autograd.set_detect_anomaly(True)

def NIFTY_loss(z1: torch.Tensor, z2: torch.Tensor, projection_layer1:nn.Module, projection_layer2:nn.Module):
    
    def D(x1, x2):  # negative cosine similarity
        return -F.cosine_similarity(x1, x2.detach(), dim=-1).mean()

    # projector
    p1 = projection_layer1(z1)
    p2 = projection_layer1(z2)

    # predictor
    h1 = projection_layer2(p1)
    h2 = projection_layer2(p2)

    l1 = D(h1, p2) / 2
    l2 = D(h2, p1) / 2

    return (l1 + l2)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    
def update_bestresults(results, best_results):
    best = {}
    if best_results is None:
        return results

    else:
        for key, result in results.items():
            best[key] = max(result, best_results[key])
    return best


def train(model,
          fair_model,
          predictor,
          data,
          split_edge,
          optimizer,
          fair_optimizer,
          batch_size,
          reg_lambda,
          no_intervention,
          link_level,
          maskinput: bool = True,
          cnprobs: Iterable[float]=[],
          alpha: float=None,
          nifty_loss: bool=False,
          projection_layer: nn.Module=None, alternate_training=0):
    

    
    if alpha is not None:
        predictor.setalpha(alpha)
        
    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    pos_train_edge = pos_train_edge.t()

    total_loss = []
    total_fair_loss = []
    adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)
            
    negedge = split_edge['train']['edge_neg'].to(data.x.device).t()
    
    # for perm in [list(PermIterator(adjmask.device, adjmask.shape[0], min(adjmask.shape[0], batch_size)))[0]]:
    for perm in PermIterator(adjmask.device, adjmask.shape[0], min(adjmask.shape[0], batch_size)):

        optimizer.zero_grad()
        fair_optimizer.zero_grad()
        
        if maskinput:
            adjmask[perm] = 0
            tei = pos_train_edge[:, adjmask]
            adj = SparseTensor.from_edge_index(tei,
                               sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(
                                   pos_train_edge.device, non_blocking=True)
            adjmask[perm] = 1
            adj = adj.to_symmetric()
        else:
            adj = data.adj_t
        
        model = model.to(data.x.device)
        h = model(data.x, adj)
        
        if nifty_loss:
            flipped_x = data.x
            flipped_x[:, -1] = torch.abs(1 - flipped_x[:, -1])
            h_flipped = model(flipped_x, adj)
        
        edge = pos_train_edge[:, perm]
        
        if not link_level:
            pos_outs = predictor.multidomainforward(h,
                                                    adj,
                                                    edge,
                                                    cndropprobs=cnprobs)
            edge = negedge[:, perm]
            neg_outs = predictor.multidomainforward(h, adj, edge, cndropprobs=cnprobs)
            
            if nifty_loss:
                pos_outs_flipped = predictor.multidomainforward(h_flipped,
                                                    adj,
                                                    edge,
                                                    cndropprobs=cnprobs)
                edge = negedge[:, perm]
                neg_outs_flipped = predictor.multidomainforward(h_flipped, adj, edge, cndropprobs=cnprobs)
                
        else:
            pos_outs, pos_embs = predictor.multidomainforward(h,
                                                    adj,
                                                    edge,
                                                    cndropprobs=cnprobs)
            edge = negedge[:, perm]
            neg_outs, neg_embs = predictor.multidomainforward(h, adj, edge, cndropprobs=cnprobs)
            
            if nifty_loss:
                pos_outs_flipped, pos_embs_flipped = predictor.multidomainforward(h_flipped,
                                                    adj,
                                                    edge,
                                                    cndropprobs=cnprobs)
                edge = negedge[:, perm]
                neg_outs_flipped, neg_embs_flipped = predictor.multidomainforward(h_flipped, adj, edge, cndropprobs=cnprobs)

        pos_losss = -F.logsigmoid(pos_outs).mean()
        neg_losss = -F.logsigmoid(-neg_outs).mean()
            
        if not no_intervention:
            
            if nifty_loss:
                if link_level:
                    z1 = torch.cat([pos_embs, neg_embs])
                    z2 = torch.cat([pos_embs_flipped, neg_embs_flipped])
                    fairness_loss = NIFTY_loss(z1, z2, projection_layer[0], projection_layer[1])
                    
                else:
                    z1 = h
                    z2 = h_flipped
                    fairness_loss = NIFTY_loss(z1, z2, projection_layer[0], projection_layer[1])

                loss = neg_losss + pos_losss - (reg_lambda * fairness_loss)
                
                total_fair_loss.append(fairness_loss.item())
                
                loss.backward()
                optimizer.step()
            
            else:

                if link_level:
                    protected_groups_labels_pos = (data.y[pos_train_edge[:, perm]].sum(0).long() == 1).float()
                    protected_groups_labels_neg = (data.y[negedge[:, perm]].sum(0).long() == 1).float()
                    
                    embs = torch.cat([pos_embs, neg_embs])
                    labels = torch.cat([protected_groups_labels_pos, protected_groups_labels_neg])
                    labels = torch.nn.functional.one_hot(labels.long(), num_classes=len(labels.unique())).float()
                    
                    # Following Tip #1 from https://github.com/soumith/ganhacks?tab=readme-ov-file
                    embs = F.tanh(embs)
                    
                    
                    fair_pred = fair_model(embs)
                    
                    if args.laftr_dp:
                        sens_groups = torch.cat([protected_groups_labels_pos, protected_groups_labels_neg])
                        sens_preds = torch.cat([pos_fair_outs, neg_fair_outs]).flatten()
                        sens_group_proportion = 1 / sens_groups.sum()
                        non_sensitive_group_proportion = 1 - sens_group_proportion
                        fairness_loss = 1 - (sens_group_proportion*torch.abs(sens_preds[sens_groups.bool()] - sens_groups[sens_groups.bool()])).sum() + (non_sensitive_group_proportion*torch.abs(sens_preds[~sens_groups.bool()] - sens_groups[~sens_groups.bool()])).sum()

                    elif args.laftr_eo:
                        # pos_fair_outs = pos_fair_outs.flatten()
                        # sens_group_pos = protected_groups_labels_pos[protected_groups_labels_pos == 1]
                        # non_sens_group_pos = protected_groups_labels_pos[protected_groups_labels_pos == 0]
                        # sens_group_pos_proportion = 1 / sens_group_pos.shape[0]
                        # non_sens_group_pos_proportion = 1 / non_sens_group_pos.shape[0]
                        # fairness_loss = 2 - (sens_group_pos_proportion * torch.abs(pos_fair_outs[protected_groups_labels_pos.bool()] - sens_group_pos)).sum() + (non_sens_group_pos_proportion * torch.abs(pos_fair_outs[~protected_groups_labels_pos.bool()] - non_sens_group_pos)).sum()
                        
                        # import code
                        # code.interact(local={**locals(), **globals()})
                        
                        h_f_a = fair_pred.softmax(dim=-1)
                        
                        # protected_groups_labels_pos = (data.y[split_edge['train']['edge']].sum(1).long() == 1).float()
                        # pos_sens_group_proportion = 1.0 / protected_groups_labels_pos.sum()
                        # pos_non_sensitive_group_proportion = 1.0 / (protected_groups_labels_pos == 0).sum()
                        
                        # Extract the embeddings and labels only for the positive edges
                        # since only the positive edges matter for Equality of Opportunity
                        pos_pairs_fair_outputs = h_f_a[:pos_embs.shape[0], :]
                        pos_pairs_fair_labels = labels[:pos_embs.shape[0]]
                        
                        # class_weights = labels[:pos_embs.shape[0]].sum(0) / labels[:pos_embs.shape[0]].sum()
                        class_weights = 1 / labels[:pos_embs.shape[0]].sum(0)

                        import code
                        code.interact(local={**locals(), **globals()})
                        # fairness_loss = 2 - ((pos_pairs_fair_outputs - pos_pairs_fair_labels).abs().sum(0) * class_weights).sum()
                        fairness_loss = 2 - ((pos_pairs_fair_outputs[:, 1] - pos_pairs_fair_labels[:, 1]).abs().sum(0) * class_weights[1]).sum()


                    else:
                        # import code
                        # code.interact(local={**locals(), **globals()})
                        fairness_loss = F.binary_cross_entropy_with_logits(fair_pred, labels)
                    
                    if is_model_trainable(fair_model):
                        fairness_loss.backward(retain_graph=True)
                    
                    if args.dp_only:
                        loss = fairness_loss
                    else:
                        loss = neg_losss + pos_losss - (reg_lambda * fairness_loss.item())
                    total_fair_loss.append(fairness_loss.item())
                    
                    if is_model_trainable(model):
                        loss.backward()
                    
                    if args.laftr_eo:
                        # For WGAN it is very common to apply the gradient clipping trick.
                        # We need this to make it 1-Lipschitz
                        torch.nn.utils.clip_grad_norm_(fair_model.parameters(),0.1)
                        # torch.nn.utils.clip_grad_norm_(predictor.parameters(),0.1)
                        # torch.nn.utils.clip_grad_norm_(model.parameters(),0.1)
                        # pass
                    
                    if is_model_trainable(fair_model):
                        fair_optimizer.step()
                        
                    if is_model_trainable(model):
                        optimizer.step()
                    
                    
                    
                else:                
                    # Following Tip #1 from https://github.com/soumith/ganhacks?tab=readme-ov-file
                    h = F.tanh(h)
                    
                    if args.fair_learner_gnn:
                        row, col, _ = adj.coo()
                        edge_index = torch.stack([row, col], dim=0)
                        fair_pred = fair_model(h, edge_index)
                    else:
                        fair_pred = fair_model(h)
                    
                    labels = data.y.long()
                    labels = torch.nn.functional.one_hot(labels, num_classes=len(labels.unique())).float()
                    
                    fairness_loss = F.binary_cross_entropy_with_logits(fair_pred.squeeze(-1), labels)
                    
                    if is_model_trainable(fair_model):
                        fairness_loss.backward(retain_graph=True)

                    if args.dp_only:
                        loss = fairness_loss
                    else:
                        loss = neg_losss + pos_losss - (reg_lambda * fairness_loss.item())
                    total_fair_loss.append(fairness_loss.item())
                    
                    if is_model_trainable(model):
                        loss.backward()
                        
                    if is_model_trainable(fair_model):
                        fair_optimizer.step()
                        
                    if is_model_trainable(model):
                        optimizer.step()
            
            
                batch_acc = binary_accuracy(torch.sigmoid(fair_pred).cpu(), labels.cpu()).item()

                # Show fair model gradients in the first layer
                if alternate_training == 0:
                    print("Gradients", fair_model.lins[0].weight.grad.norm())
                logger.info(f"{'FAIR MODEL TRAINING' if alternate_training == 0 else 'GENERATOR TRAINING'} - Train - Current fair model acc {batch_acc} - fairness_loss {fairness_loss.item()} - loss {loss.item()} - pos_loss {pos_losss.item()} - neg_loss {neg_losss.item()}")
            
                
        else:
            loss = neg_losss + pos_losss
            loss.backward()
            optimizer.step()


        total_loss.append(loss.item())
        
    total_loss = np.average(total_loss)
    total_fair_loss = np.average(total_fair_loss)
    
    return total_loss, total_fair_loss


@torch.no_grad()
def test(model, fair_model, predictor, data, split_edge, evaluator, batch_size, link_level,
         use_valedges_as_input):
    model.eval()
    predictor.eval()

    pos_train_edge = split_edge['train']['edge'].to(data.adj_t.device())
    neg_train_edge = split_edge['train']['edge_neg'].to(data.adj_t.device())
    pos_valid_edge = split_edge['valid']['edge'].to(data.adj_t.device())
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.adj_t.device())
    pos_test_edge = split_edge['test']['edge'].to(data.adj_t.device())
    neg_test_edge = split_edge['test']['edge_neg'].to(data.adj_t.device())

    adj = data.adj_t
    h = model(data.x, adj)
    
    if not link_level:
        pos_train_pred = torch.cat([
            predictor(h, adj, pos_train_edge[perm].t()).squeeze().cpu()
            for perm in PermIterator(pos_train_edge.device,
                                    pos_train_edge.shape[0], batch_size, False)
        ],
                                dim=0)
        
        
        neg_train_pred = torch.cat([
            predictor(h, adj, neg_train_edge[perm].t()).squeeze().cpu()
            for perm in PermIterator(neg_train_edge.device,
                                        neg_train_edge.shape[0], batch_size, False)
        ],
                                    dim=0)


        pos_valid_pred = torch.cat([
            predictor(h, adj, pos_valid_edge[perm].t()).squeeze().cpu()
            for perm in PermIterator(pos_valid_edge.device,
                                    pos_valid_edge.shape[0], batch_size, False)
        ],
                                dim=0)
        neg_valid_pred = torch.cat([
            predictor(h, adj, neg_valid_edge[perm].t()).squeeze().cpu()
            for perm in PermIterator(neg_valid_edge.device,
                                    neg_valid_edge.shape[0], batch_size, False)
        ],
                                dim=0)
        if use_valedges_as_input:
            adj = data.full_adj_t
            h = model(data.x, adj)

        pos_test_pred = torch.cat([
            predictor(h, adj, pos_test_edge[perm].t()).squeeze().cpu()
            for perm in PermIterator(pos_test_edge.device, pos_test_edge.shape[0],
                                    batch_size, False)
        ],
                                dim=0)

        neg_test_pred = torch.cat([
            predictor(h, adj, neg_test_edge[perm].t()).squeeze().cpu()
            for perm in PermIterator(neg_test_edge.device, neg_test_edge.shape[0],
                                    batch_size, False)
        ],
                                dim=0)
        
        if args.fair_learner_gnn:
            row, col, _ = adj.coo()
            edge_index = torch.stack([row, col], dim=0)
            fair_pred = fair_model(h, edge_index)
        else:
            fair_pred = fair_model(h)
            
        fairness_acc = binary_accuracy(fair_pred.cpu(), torch.nn.functional.one_hot(data.y.long().cpu()))
        fairness_train, fairness_test = fairness_acc, fairness_acc
        
    else:

        pos_train_pred = torch.empty(0)
        pos_train_fair_pred = torch.empty(0)
        pos_train_fair_labels = torch.empty(0)
        for perm in PermIterator(pos_train_edge.device, pos_train_edge.shape[0], batch_size, False):
            pred, pred_embs = predictor(h, adj, pos_train_edge[perm].t())
            pos_train_pred = torch.cat((pos_train_pred.squeeze().cpu(), pred.view(-1).cpu()), dim=0)
            pos_train_fair_pred = torch.cat([pos_train_fair_pred.cpu(), fair_model(pred_embs).cpu()], dim=0)
            pos_train_fair_labels = torch.cat([pos_train_fair_labels.cpu(), data.y[pos_train_edge[perm].t()].sum(0).long().cpu()], dim=0)

        neg_train_pred = torch.empty(0)
        neg_train_fair_pred = torch.empty(0)
        neg_train_fair_labels = torch.empty(0)
        for perm in PermIterator(neg_train_edge.device, neg_train_edge.shape[0], batch_size, False):
            pred, pred_embs = predictor(h, adj, neg_train_edge[perm].t())
            neg_train_pred = torch.cat((neg_train_pred.squeeze().cpu(), pred.view(-1).cpu()), dim=0)
            neg_train_fair_pred = torch.cat([neg_train_fair_pred.cpu(), fair_model(pred_embs).cpu()], dim=0)
            neg_train_fair_labels = torch.cat([neg_train_fair_labels.cpu(), data.y[neg_train_edge[perm].t()].sum(0).long().cpu()], dim=0)

        pos_valid_pred = torch.empty(0)
        pos_valid_fair_pred = torch.empty(0)
        pos_valid_fair_labels = torch.empty(0)
        for perm in PermIterator(pos_valid_edge.device, pos_valid_edge.shape[0], batch_size, False):
            pred, pred_embs = predictor(h, adj, pos_valid_edge[perm].t())
            pos_valid_pred = torch.cat((pos_valid_pred.squeeze().cpu(), pred.view(-1).cpu()), dim=0)
            pos_valid_fair_pred = torch.cat([pos_valid_fair_pred.cpu(), fair_model(pred_embs).cpu()], dim=0)
            pos_valid_fair_labels = torch.cat([pos_valid_fair_labels.cpu(), data.y[pos_train_edge[perm].t()].sum(0).long().cpu()], dim=0)

        neg_valid_pred = torch.empty(0)
        neg_valid_fair_pred = torch.empty(0)
        neg_valid_fair_labels = torch.empty(0)
        for perm in PermIterator(neg_valid_edge.device, neg_valid_edge.shape[0], batch_size, False):
            pred, pred_embs = predictor(h, adj, neg_valid_edge[perm].t())
            neg_valid_pred = torch.cat((neg_valid_pred.squeeze().cpu(), pred.view(-1).cpu()), dim=0)
            neg_valid_fair_pred = torch.cat([neg_valid_fair_pred.cpu(), fair_model(pred_embs).cpu()], dim=0)
            neg_valid_fair_labels = torch.cat([neg_valid_fair_labels.cpu(), data.y[neg_train_edge[perm].t()].sum(0).long().cpu()], dim=0)

        if use_valedges_as_input:
            adj = data.full_adj_t
            h = model(data.x, adj)

        pos_test_pred = torch.empty(0)
        pos_test_fair_pred = torch.empty(0)
        pos_test_fair_labels = torch.empty(0)
        for perm in PermIterator(pos_test_edge.device, pos_test_edge.shape[0], batch_size, False):
            pred, pred_embs = predictor(h, adj, pos_test_edge[perm].t())
            pos_test_pred = torch.cat((pos_test_pred.squeeze().cpu(), pred.view(-1).cpu()), dim=0)
            pos_test_fair_pred = torch.cat([pos_test_fair_pred.cpu(), fair_model(pred_embs).cpu()], dim=0)
            pos_test_fair_labels = torch.cat([pos_test_fair_labels.cpu(), data.y[pos_train_edge[perm].t()].sum(0).long().cpu()], dim=0)

        neg_test_pred = torch.empty(0)
        neg_test_fair_pred = torch.empty(0)
        neg_test_fair_labels = torch.empty(0)
        for perm in PermIterator(neg_test_edge.device, neg_test_edge.shape[0], batch_size, False):
            pred, pred_embs = predictor(h, adj, neg_test_edge[perm].t())
            neg_test_pred = torch.cat((neg_test_pred.squeeze().cpu(), pred.view(-1).cpu()), dim=0)
            neg_test_fair_pred = torch.cat([neg_test_fair_pred.cpu(), fair_model(pred_embs).cpu()], dim=0)
            neg_test_fair_labels = torch.cat([neg_test_fair_labels.cpu(), data.y[neg_train_edge[perm].t()].sum(0).long().cpu()], dim=0)
            
        
        fair_train_preds, fair_train_labels = torch.cat([pos_train_fair_pred, neg_train_fair_pred]), torch.nn.functional.one_hot(torch.cat([(pos_train_fair_labels == 1).long(), (neg_train_fair_labels == 1).long()]).long())
        fair_valid_preds, fair_valid_labels = torch.cat([pos_valid_fair_pred, neg_valid_fair_pred]), torch.nn.functional.one_hot(torch.cat([(pos_valid_fair_labels == 1).long(), (neg_valid_fair_labels == 1).long()]).long())
        fair_test_preds, fair_test_labels = torch.cat([pos_test_fair_pred, neg_test_fair_pred]), torch.nn.functional.one_hot(torch.cat([(pos_test_fair_labels == 1).long(), (neg_test_fair_labels == 1).long()]).long())
        # import code
        # code.interact(local={**locals(), **globals()})
        fairness_train = binary_accuracy(torch.sigmoid(fair_train_preds).cpu(), fair_train_labels.cpu())
        fairness_valid = binary_accuracy(torch.sigmoid(fair_valid_preds).cpu(), fair_valid_labels.cpu())
        fairness_test = binary_accuracy(torch.sigmoid(fair_test_preds).cpu(), fair_test_labels.cpu())
    

    train_edges = torch.cat([pos_train_edge.t(), neg_train_edge.t()], dim=1)
    train_preds = torch.cat([pos_train_pred, neg_train_pred]).to(train_edges.device)
    train_labels = torch.cat([torch.ones(pos_train_edge.size(0)), torch.zeros(neg_train_edge.size(0))]).to(train_edges.device)
    
    val_edges = torch.cat([pos_valid_edge.t(), neg_valid_edge.t()], dim=1)
    val_preds = torch.cat([pos_valid_pred, neg_valid_pred]).to(val_edges.device)
    val_labels = torch.cat([torch.ones(pos_valid_edge.size(0)), torch.zeros(neg_valid_edge.size(0))]).to(val_edges.device)
    
    test_edges = torch.cat([pos_test_edge.t(), neg_test_edge.t()], dim=1)
    test_preds = torch.cat([pos_test_pred, neg_test_pred]).to(test_edges.device)
    test_labels = torch.cat([torch.ones(pos_test_edge.size(0)), torch.zeros(neg_test_edge.size(0))]).to(test_edges.device)
    
    train_protected_groups_labels = (data.y[train_edges].sum(0).long() == 1).float().to(train_edges.device)
    val_protected_groups_labels = (data.y[val_edges].sum(0).long() == 1).float().to(train_edges.device)
    test_protected_groups_labels = (data.y[test_edges].sum(0).long() == 1).float().to(train_edges.device)
    
    train_tprd_no_abs = true_positive_rate_disparity(train_labels, train_preds, train_protected_groups_labels.detach())
    val_tprd_no_abs = true_positive_rate_disparity(val_labels, val_preds, val_protected_groups_labels.detach())
    test_tprd_no_abs = true_positive_rate_disparity(test_labels, test_preds, test_protected_groups_labels.detach())
    
    train_prd_no_abs = positive_rate_disparity(train_preds, train_protected_groups_labels.detach())
    val_prd_no_abs = positive_rate_disparity(val_preds, val_protected_groups_labels.detach())
    test_prd_no_abs = positive_rate_disparity(test_preds, test_protected_groups_labels.detach())
    
    train_tprd = train_tprd_no_abs.abs()
    val_tprd = val_tprd_no_abs.abs()
    test_tprd = test_tprd_no_abs.abs()
    
    train_prd = train_prd_no_abs.abs()
    val_prd = val_prd_no_abs.abs()
    test_prd = test_prd_no_abs.abs()
    
    train_protected_groups = (data.y[train_edges].sum(0).long()).float().to(train_edges.device)
    val_protected_groups = (data.y[val_edges].sum(0).long()).float().to(train_edges.device)
    test_protected_groups = (data.y[test_edges].sum(0).long()).float().to(train_edges.device)
    
    train_prd_p_mm, train_prd_p_mf, train_prd_p_ff = group_positive_rate_disparity(train_preds, train_protected_groups.detach())
    val_prd_p_mm, val_prd_p_mf, val_prd_p_ff = group_positive_rate_disparity(val_preds, val_protected_groups.detach())
    test_prd_p_mm, test_prd_p_mf, test_prd_p_ff = group_positive_rate_disparity(test_preds, test_protected_groups.detach())

    train_tprd_p_mm, train_tprd_p_mf, train_tprd_p_ff = group_true_positive_rate_disparity(train_labels, train_preds, train_protected_groups.detach())
    val_tprd_p_mm, val_tprd_p_mf, val_tprd_p_ff = group_true_positive_rate_disparity(val_labels, val_preds, val_protected_groups.detach())
    test_tprd_p_mm, test_tprd_p_mf, test_tprd_p_ff = group_true_positive_rate_disparity(test_labels, test_preds, test_protected_groups.detach())
    
    saved_output = {
        'train_preds': train_preds,
        'train_true': train_labels,
        'train_protected_groups': train_protected_groups_labels,
        'val_preds': val_preds,
        'val_true': val_labels,
        'val_protected_groups': val_protected_groups_labels,
        'test_preds': test_preds,
        'test_true': test_labels,
        'test_protected_groups': test_protected_groups_labels
    }

    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']

        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']
        
        # top_k_pairs = torch.argsort(train_preds)[:K]
        # results[f'rep{0}_TrainSens@{K}'] = train_protected_groups_labels[top_k_pairs].sum() / train_protected_groups_labels.sum()
        # results[f'rep{0}_ValidSens@{K}'] = val_protected_groups_labels[top_k_pairs].sum() / val_protected_groups_labels.sum()
        # results[f'rep{0}_TestSens@{K}'] = test_protected_groups_labels[top_k_pairs].sum() / test_protected_groups_labels.sum()
        
        results[f'rep{0}_TrainHits@{K}'] = train_hits
        results[f'rep{0}_ValHits@{K}'] = valid_hits
        results[f'rep{0}_TestHits@{K}'] = test_hits
        
    results[f'rep{0}_true_positive_rate_disparity_Train'] = train_tprd.item()
    results[f'rep{0}_true_positive_rate_disparity_Valid'] = val_tprd.item()
    results[f'rep{0}_true_positive_rate_disparity_Test'] = test_tprd.item()
    
    results[f'rep{0}_positive_rate_disparity_Train'] = train_prd.item()
    results[f'rep{0}_positive_rate_disparity_Valid'] = val_prd.item()
    results[f'rep{0}_positive_rate_disparity_Test'] = test_prd.item()
    
    results[f'rep{0}_true_positive_rate_disparity_Train_no_abs'] = train_tprd_no_abs
    results[f'rep{0}_true_positive_rate_disparity_Valid_no_abs'] = val_tprd_no_abs
    results[f'rep{0}_true_positive_rate_disparity_Test_no_abs'] = test_tprd_no_abs

    results[f'rep{0}_positive_rate_disparity_Train_no_abs'] = train_prd_no_abs
    results[f'rep{0}_positive_rate_disparity_Valid_no_abs'] = val_prd_no_abs
    results[f'rep{0}_positive_rate_disparity_Test_no_abs'] = test_prd_no_abs
    
    results[f'rep{0}_group_positive_rate_disparity_Train_p_mm'] = train_prd_p_mm
    results[f'rep{0}_group_positive_rate_disparity_Train_p_mf'] = train_prd_p_mf
    results[f'rep{0}_group_positive_rate_disparity_Train_p_ff'] = train_prd_p_ff
    results[f'rep{0}_group_positive_rate_disparity_Valid_p_mm'] = val_prd_p_mm
    results[f'rep{0}_group_positive_rate_disparity_Valid_p_mf'] = val_prd_p_mf
    results[f'rep{0}_group_positive_rate_disparity_Valid_p_ff'] = val_prd_p_ff
    results[f'rep{0}_group_positive_rate_disparity_Test_p_mm']  = test_prd_p_mm
    results[f'rep{0}_group_positive_rate_disparity_Test_p_mf']  = test_prd_p_mf
    results[f'rep{0}_group_positive_rate_disparity_Test_p_ff']  = test_prd_p_ff
    
    results[f'rep{0}_group_true_positive_rate_disparity_Train_p_mm'] = train_tprd_p_mm
    results[f'rep{0}_group_true_positive_rate_disparity_Train_p_mf'] = train_tprd_p_mf
    results[f'rep{0}_group_true_positive_rate_disparity_Train_p_ff'] = train_tprd_p_ff
    results[f'rep{0}_group_true_positive_rate_disparity_Valid_p_mm'] = val_tprd_p_mm
    results[f'rep{0}_group_true_positive_rate_disparity_Valid_p_mf'] = val_tprd_p_mf
    results[f'rep{0}_group_true_positive_rate_disparity_Valid_p_ff'] = val_tprd_p_ff
    results[f'rep{0}_group_true_positive_rate_disparity_Test_p_mm']  = test_tprd_p_mm
    results[f'rep{0}_group_true_positive_rate_disparity_Test_p_mf']  = test_tprd_p_mf
    results[f'rep{0}_group_true_positive_rate_disparity_Test_p_ff']  = test_tprd_p_ff
    
    results[f'rep{0}_adv_acc_train'] = fairness_train.item()
    results[f'rep{0}_adv_acc'] = fairness_test.item()
    
    return results, h.cpu(), saved_output


@logger.catch
def main():
    print(args, flush=True)
    
    # Trying our embeddings with GAE.
    gae_embedding_experiment = None
    if '_gae' in args.dataset:
        gae_embedding_experiment = True
        args.dataset = args.dataset.replace('_gae', '')
    
    dataset_paths = {
        'facebook': '/home/jrm28/fairness/data/graphs/facebook.pt',
        'facebook_graphair': "/home/jrm28/fairness/graphair/fairgraph/method/checkpoint/out/AUGMENTED_facebook_10000_epochs_2024-03-13_14-50-37/splits.pt",
        'facebook_nifty': "/home/jrm28/fairness/NeuralCommonNeighbor/dataset/splits/facebook_nifty.pt",
        'facebook_edits': "/home/jrm28/fairness/NeuralCommonNeighbor/dataset/splits/facebook_edits.pt",
        'facebook_gae':'/home/jrm28/fairness/NeuralCommonNeighbor/gemb/NCN_facebook_facebook_puregcn_cn1_256.pt',
        'nba': "/home/jrm28/fairness/data/graphs/nba.pt",
        'pokec_n': "/home/jrm28/fairness/data/graphs/pokec_n.pt",
        'pokec_z': "/home/jrm28/fairness/data/graphs/pokec_z.pt",
        'credit': "/home/jrm28/fairness/data/graphs/credit.pt",
        'gplus': '/home/jrm28/fairness/subgraph_sketching-original/dataset/gplus/processed/gplus_100129275726588145876.pt',
        'sbm': '/home/jrm28/fairness/subgraph_sketching-original/dataset/sbm/processed/sbm.pt',
        'sbm_medium': '/home/jrm28/fairness/subgraph_sketching-original/dataset/sbm/processed/sbm_medium.pt',
        'sbm_bigger': '/home/jrm28/fairness/subgraph_sketching-original/dataset/sbm/processed/sbm_bigger.pt',
    }
    
    dataset_path = dataset_paths[args.dataset]

    hpstr = str(args).replace(" ", "").replace("Namespace(", "").replace(
        ")", "").replace("True", "1").replace("False", "0").replace("=", "").replace("epochs", "").replace("runs", "").replace("save_gemb", "")
    writer = SummaryWriter(f"./rec/{args.model}_{args.predictor}")
    writer.add_text("hyperparams", hpstr)

    evaluator = Evaluator(name=f'ogbl-ppa')

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    # device = torch.cuda.set_device(0)
    # device='cpu'
    
    dataset_file = f"/home/jrm28/fairness/NeuralCommonNeighbor/dataset/splits/{args.dataset}{'_node_split' if args.node_split else ''}.pt" 
    if os.path.isfile(dataset_file):
        logger.info("Loading dataset from disk...")
        data, split_edge = torch.load(dataset_file)
    else:
        logger.info("Sampling dataset and saving to disk...")
        data, split_edge = loaddataset(args.dataset, dataset_path, args.use_valedges_as_input, args.load, args)
        torch.save((data, split_edge), dataset_file)
        
        
    if gae_embedding_experiment:
        dataset_paths_gae = {
            'facebook': "/home/jrm28/fairness/NeuralCommonNeighbor/gemb/NCN_facebook_facebook_puregcn_cn1_256.pt",
        }[args.dataset]
        data.x = torch.load(dataset_paths_gae)
        
        
    if args.fairsample:
        if os.path.isfile(f'/home/jrm28/fairness/NeuralCommonNeighbor/dataset/splits/fairsample_{args.dataset}.pt'):
            logger.info("Loading fair samples...")
            data, split_edge = torch.load(f'/home/jrm28/fairness/NeuralCommonNeighbor/dataset/splits/fairsample_{args.dataset}.pt')
        else:
            logger.info("Sampling fair samples...")
            data, split_edge = fairsample(data, split_edge)
            torch.save((data, split_edge), f'/home/jrm28/fairness/NeuralCommonNeighbor/dataset/splits/fairsample_{args.dataset}.pt')
        
    
    projection_layers = None
    if args.nifty:
        # Adding the sensitive feature as the last one in the tensor
        data.x = torch.cat([data.x.to(device), data.y[None, :].to(device).t()], dim=1)
        projection_layer1 = torch.nn.Linear(args.hiddim, args.hiddim).to(device=device)
        projection_layer2 = torch.nn.Linear(args.hiddim, args.hiddim).to(device=device)
        projection_layers = (projection_layer1, projection_layer2)

    
    data = data.to(device)

    predictor_name = args.predictor 
    predictor_name += "_mod" if args.link_level else ""

    predfn = predictor_dict[predictor_name]
    if args.predictor != "cn0":
        predfn = partial(predfn, cndeg=args.cndeg)
    if args.predictor in ["cn1", "cn1_mod", "incn1cn1", "scn1", "catscn1", "sincn1cn1"]:
        predfn = partial(predfn, use_xlin=args.use_xlin, tailact=args.tailact, twolayerlin=args.twolayerlin, beta=args.beta)
    if args.predictor == "incn1cn1":
        predfn = partial(predfn, depth=args.depth, splitsize=args.splitsize, scale=args.probscale, offset=args.proboffset, trainresdeg=args.trndeg, testresdeg=args.tstdeg, pt=args.pt, learnablept=args.learnpt, alpha=args.alpha)
    
    ret = []

    for run in range(0, args.runs):
        
        run_name = '[No Intervention] - ' if args.no_intervention else ''
        run_name += '_GNN_FAIR_' if args.fair_learner_gnn else ''
        run_name += 'FairSample_' if args.fairsample else ''
        run_name += 'DP_only - ' if args.dp_only else ''
        run_name += 'FAIR_EMBS - ' if gae_embedding_experiment else ''
        run_name += 'NIFTY_LOSS - ' if args.nifty else ''
        # run_name += f'{"NCN" if args.predictor == "cn1" else "NCNC"}_{args.dataset}'
        if args.predictor == "cn1":
            run_name += f'{"NCN"}_{args.dataset}'
        elif args.predictor == "cn0":
            run_name += f'{"GAE"}_{args.dataset}'
        else:
            run_name += f'{"NCNC"}_{args.dataset}'
            
        run_name += "_link_level" if args.link_level else ""
        run_name += "_node_split" if args.node_split else ""
        run_name += "_laftr_dp" if "_link_level" in run_name and args.laftr_dp else ""
        run_name += "_laftr_eo" if "_link_level" in run_name and args.laftr_eo else ""
        
        if args.dataset == 'facebook':
            run_name = "NEW_SPLITS" + run_name
        wandb_run = wandb.init(project="lpfairness", entity="joaopedromattos", config=args, name=run_name, mode="online" if not args.no_wandb else "disabled", tags=['tunning'] if args.wandb_sweep else [])

        artifact = wandb.Artifact(args.dataset, type="dataset")
        artifact.add_reference(f"file:///{dataset_path}")
        
        wandb_run.use_artifact(artifact)
        
        if args.wandb_sweep:
            args.adv_lr = wandb.config.adv_lr
            args.reg_lambda = wandb.config.reg_lambda

        
        set_seed(run)

        bestscore = None
        
        # build model
        model = GCN(data.num_features, args.hiddim, args.hiddim, args.mplayers,
                    args.gnndp, args.ln, args.res, data.max_x,
                    args.model, args.jk, args.gnnedp,  xdropout=args.xdp, taildropout=args.tdp, noinputlin=args.loadx).to(device)
        
        fair_model_class = FairLearner_GNN if args.fair_learner_gnn else FairLearner
        # fair_model = fair_model_class(args.hiddim, args.hiddim * 2, 2, out_channels=2).to(device)
        fair_model = fair_model_class(args.hiddim, args.hiddim * 2, 4, out_channels=2).to(device)
        
        wandb.watch(model, log="all")
        wandb.watch(fair_model, log="all")
        
        if args.loadx:
            with torch.no_grad():
                model.xemb[0].weight.copy_(torch.load(f"gemb/{args.dataset}_{args.model}_cn1_{args.hiddim}_{run}.pt", map_location="cpu"))
            model.xemb[0].weight.requires_grad_(False)
            
        predictor = predfn(args.hiddim, args.hiddim, 1, args.nnlayers,
                        args.predp, args.preedp, args.lnnn).to(device)
        
        if args.loadmod:
            keys = model.load_state_dict(torch.load(f"gmodel/{args.dataset}_{args.model}_cn1_{args.hiddim}_{run}.pt", map_location="cpu"), strict=False)
            print("unmatched params", keys, flush=True)
            keys = predictor.load_state_dict(torch.load(f"gmodel/{args.dataset}_{args.model}_cn1_{args.hiddim}_{run}.pre.pt", map_location="cpu"), strict=False)
            print("unmatched params", keys, flush=True)
        

        optimizer = torch.optim.Adam([{'params': model.parameters(), "lr": args.gnnlr}, 
        {'params': predictor.parameters(), 'lr': args.prelr}])
        
        fair_optimizer = torch.optim.Adam(fair_model.parameters(), lr=args.adv_lr)
        
        model = model.to(device)
        fair_model = fair_model.to(device)
        predictor = predictor.to(device)
        alternate_training = 0 # Controls whether we are training the fair model or the predictor model.
        for epoch in range(1, 1 + args.epochs):
            alpha = max(0, min((epoch-5)*0.1, 1)) if args.increasealpha else None
            t1 = time.time()
            print("training")
            
            if epoch % 10 == 0 and args.alternate_training:        
                alternate_training = 1 - alternate_training
                if alternate_training == 1:
                    disable_training(fair_model)
                    enable_training(predictor)
                    enable_training(model)
                    
                    model.train()
                    predictor.train()
                    fair_model.eval()
                else:
                    enable_training(fair_model)
                    disable_training(model)
                    disable_training(predictor)
                    
                    model.eval()
                    predictor.eval()
                    fair_model.train()
            else:
                model.train()
                predictor.train()
                fair_model.train()
                    
            
            loss, fair_loss = train(model, fair_model, predictor, data, split_edge, optimizer, fair_optimizer,
                        args.batch_size, args.reg_lambda, args.no_intervention, args.link_level, args.maskinput, [], alpha, args.nifty, projection_layers, alternate_training)
            print("after training")
            print(f"trn time {time.time()-t1:.2f} s", flush=True)
            if True:
                t1 = time.time()
                results, h, saved_output = test(model, fair_model, predictor, data, split_edge, evaluator,
                            args.testbs, args.link_level, args.use_valedges_as_input)
                print(f"test time {time.time()-t1:.2f} s")
                
                # print(results, flush=True)
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                torch.save(saved_output, f"/home/jrm28/fairness/NeuralCommonNeighbor/saved_output/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}_{epoch}{timestamp}.pt")

                if True:
                    
                    if bestscore is None or bestscore[f"rep{0}_ValHits@{100}"] < results[f"rep{0}_ValHits@{100}"]:
                        train_hits, valid_hits, test_hits = results[f'rep{0}_TrainHits@{100}'], results[f'rep{0}_ValHits@{100}'], results[f'rep{0}_TestHits@{100}']
                        if args.save_gemb:
                            torch.save(h, f"/home/jrm28/fairness/NeuralCommonNeighbor/gemb/{run_name.replace(' ', '_')}_{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}.pt")
                        if args.savex:
                            torch.save(model.xemb[0].weight.detach(), f"/home/jrm28/fairness/NeuralCommonNeighbor/gemb/{run_name.replace(' ', '_')}_{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pt")
                        if args.savemod:
                            torch.save(model.state_dict(), f"/home/jrm28/fairness/NeuralCommonNeighbor/gmodel/{run_name.replace(' ', '_')}_{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pt")
                            torch.save(predictor.state_dict(), f"/home/jrm28/fairness/NeuralCommonNeighbor/gmodel/{run_name.replace(' ', '_')}_{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pre.pt")
                    
                    print(f'Run: {run + 1:02d}, '
                            f'Epoch: {epoch:02d}, '
                            f'Loss: {loss:.4f}, '
                            f'Fair Loss: {fair_loss:.4f}, '
                            f'Train: {100 * train_hits:.2f}%, '
                            f'Valid: {100 * valid_hits:.2f}%, '
                            f'Test: {100 * test_hits:.2f}%')
                print('---', flush=True)
                
                bestscore = update_bestresults(results, bestscore)
                        
            wandb.log({
                "loss": loss,
                "fair_loss": fair_loss,
                "rep0_TrainHits@100": results[f'rep{0}_TrainHits@{100}'],
                "rep0_ValHits@100": results[f'rep{0}_ValHits@{100}'],
                "rep0_TestHits@100": results[f'rep{0}_TestHits@{100}'],
                # "rep0_TrainSens@100": results[f'rep{0}_TrainSens@{100}'],
                # "rep0_ValidSens@100": results[f'rep{0}_ValidSens@{100}'],
                # "rep0_TestSens@100": results[f'rep{0}_TestSens@{100}'],
                "rep0_true_positive_rate_disparity_Train": results["rep0_true_positive_rate_disparity_Train"],
                "rep0_true_positive_rate_disparity_Valid": results["rep0_true_positive_rate_disparity_Valid"],
                "rep0_true_positive_rate_disparity_Test": results["rep0_true_positive_rate_disparity_Test"],
                "rep0_positive_rate_disparity_Train": results["rep0_positive_rate_disparity_Train"],
                "rep0_positive_rate_disparity_Valid": results["rep0_positive_rate_disparity_Valid"],
                "rep0_positive_rate_disparity_Test": results["rep0_positive_rate_disparity_Test"],
                "rep0_true_positive_rate_disparity_Train_no_abs": results[f'rep{0}_true_positive_rate_disparity_Train_no_abs'],
                "rep0_true_positive_rate_disparity_Valid_no_abs": results[f'rep{0}_true_positive_rate_disparity_Valid_no_abs'],
                "rep0_true_positive_rate_disparity_Test_no_abs": results[f'rep{0}_true_positive_rate_disparity_Test_no_abs'],
                "rep0_positive_rate_disparity_Train_no_abs": results[f'rep{0}_positive_rate_disparity_Train_no_abs'],
                "rep0_positive_rate_disparity_Valid_no_abs": results[f'rep{0}_positive_rate_disparity_Valid_no_abs'],
                "rep0_positive_rate_disparity_Test_no_abs": results[f'rep{0}_positive_rate_disparity_Test_no_abs'],
                f'rep{0}_group_positive_rate_disparity_Train_p_mm': results[f'rep{0}_group_positive_rate_disparity_Train_p_mm'],
                f'rep{0}_group_positive_rate_disparity_Train_p_mf': results[f'rep{0}_group_positive_rate_disparity_Train_p_mf'],
                f'rep{0}_group_positive_rate_disparity_Train_p_ff': results[f'rep{0}_group_positive_rate_disparity_Train_p_ff'],
                f'rep{0}_group_positive_rate_disparity_Valid_p_mm': results[f'rep{0}_group_positive_rate_disparity_Valid_p_mm'],
                f'rep{0}_group_positive_rate_disparity_Valid_p_mf': results[f'rep{0}_group_positive_rate_disparity_Valid_p_mf'],
                f'rep{0}_group_positive_rate_disparity_Valid_p_ff': results[f'rep{0}_group_positive_rate_disparity_Valid_p_ff'],
                f'rep{0}_group_positive_rate_disparity_Test_p_mm': results[f'rep{0}_group_positive_rate_disparity_Test_p_mm'] ,
                f'rep{0}_group_positive_rate_disparity_Test_p_mf': results[f'rep{0}_group_positive_rate_disparity_Test_p_mf'] ,
                f'rep{0}_group_positive_rate_disparity_Test_p_ff': results[f'rep{0}_group_positive_rate_disparity_Test_p_ff'] ,
                f'rep{0}_group_true_positive_rate_disparity_Train_p_mm': results[f'rep{0}_group_true_positive_rate_disparity_Train_p_mm'],
                f'rep{0}_group_true_positive_rate_disparity_Train_p_mf': results[f'rep{0}_group_true_positive_rate_disparity_Train_p_mf'],
                f'rep{0}_group_true_positive_rate_disparity_Train_p_ff': results[f'rep{0}_group_true_positive_rate_disparity_Train_p_ff'],
                f'rep{0}_group_true_positive_rate_disparity_Valid_p_mm': results[f'rep{0}_group_true_positive_rate_disparity_Valid_p_mm'],
                f'rep{0}_group_true_positive_rate_disparity_Valid_p_mf': results[f'rep{0}_group_true_positive_rate_disparity_Valid_p_mf'],
                f'rep{0}_group_true_positive_rate_disparity_Valid_p_ff': results[f'rep{0}_group_true_positive_rate_disparity_Valid_p_ff'],
                f'rep{0}_group_true_positive_rate_disparity_Test_p_mm': results[f'rep{0}_group_true_positive_rate_disparity_Test_p_mm'] ,
                f'rep{0}_group_true_positive_rate_disparity_Test_p_mf': results[f'rep{0}_group_true_positive_rate_disparity_Test_p_mf'] ,
                f'rep{0}_group_true_positive_rate_disparity_Test_p_ff': results[f'rep{0}_group_true_positive_rate_disparity_Test_p_ff'] ,
                f'rep{0}_adv_acc_train': results[f'rep{0}_adv_acc_train'],
                f'rep{0}_adv_acc': results[f'rep{0}_adv_acc'],
                'alternate_training': alternate_training,
                "epoch_step" : epoch - 1,
            })
        
        if args.node_classification:
            
            
            if Path(f"/home/jrm28/fairness/NeuralCommonNeighbor/gemb/{run_name.replace(' ', '_')}_{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}.pt").exists():
                logger.info("Loading embeddings from disk...")
                h = torch.load(f"/home/jrm28/fairness/NeuralCommonNeighbor/gemb/{run_name.replace(' ', '_')}_{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}.pt")
            
            if Path(f"/home/jrm28/fairness/NeuralCommonNeighbor/gmodel/{run_name.replace(' ', '_')}_{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pt").exists():
                logger.info("Loading model from disk...")
                model.load_state_dict(torch.load(f"/home/jrm28/fairness/NeuralCommonNeighbor/gmodel/{run_name.replace(' ', '_')}_{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pt"))
                
            if Path(f"/home/jrm28/fairness/NeuralCommonNeighbor/gmodel/{run_name.replace(' ', '_')}_{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pre.pt").exists():
                logger.info("Loading predictor from disk...")
                predictor.load_state_dict(torch.load(f"/home/jrm28/fairness/NeuralCommonNeighbor/gmodel/{run_name.replace(' ', '_')}_{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pre.pt"))
            
            # import code
            # code.interact(local={**locals(), **globals()})
            node_classifier = NodeClassifierModule(input_dim=h.size(1), hidden_dim=args.hiddim, output_dim=len(data.labels.unique()), num_layers=args.node_classifier_num_layers, dropout=0.0).to(device)
            node_classifier_optimizer = torch.optim.Adam(node_classifier.parameters(), lr=0.001)
            
            transform = RandomNodeSplit(num_val=0.1, num_test=0.2)
            data = transform(data)
            
            # Some datasets have labels -1, 0, 1
            if data.labels.min() < 0:
                data.labels += 1
            
            labels = torch.nn.functional.one_hot(data.labels.to(device), num_classes=len(data.labels.unique())).float()
            
            node_classifier.fit(h[data.train_mask].to(device), labels[data.train_mask, :], node_classifier_optimizer, criterion=torch.nn.CrossEntropyLoss(), epochs=150, log=wandb)
            
            y_pred = node_classifier.predict(h[data.test_mask].to(device))
            
            if len(data.labels.unique()) > 2:
                task = 'multiclass'
                result = ((y_pred.argmax(1) == labels[data.test_mask, :].argmax(1)).sum() / y_pred.shape[0]).item()
            else:
                task = 'binary'
                result = node_classifier.evaluation(y_pred, labels[data.test_mask, :], accuracy, task=task, num_classes=len(data.labels.unique()))
            
            
            
            # Save results
            torch.save((y_pred, labels[data.test_mask, :]), f"/home/jrm28/fairness/NeuralCommonNeighbor/saved_output/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}_node_classification.pt")
            
            wandb.log({"node_classification_accuracy": result})
            
        ret.append(bestscore)
        
        wandb.finish()
            

if __name__ == "__main__":
    global args 
    args = parseargs()
    
    if args.wandb_sweep:
        
        sweep_configuration = {
            "name": args.dataset,
            "metric": {"name": f'rep0_true_positive_rate_disparity_Valid', "goal": "minimize"},
            "method": "bayes",
            "parameters": {
                'adv_lr':{'max':0.01, 'min':0.0001},
                'reg_lambda':{'max':0.1, 'min':0.0001},
                'batch_size': {"max": 8192, "min": 1024},
            },
        }

        sweep_id = wandb.sweep(sweep_configuration, project="lpfairness", entity="joaopedromattos")
        
        wandb.agent(sweep_id, function=main)
    else:
        main()
