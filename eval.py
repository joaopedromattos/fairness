import torch
import os
from ast import literal_eval
from math import ceil
from sklearn.metrics import average_precision_score
from torchmetrics.functional.classification import binary_auroc

# import util
# from gelato import Gelato
import code

def normalized_precision_at_k(eval_true, eval_pred, k):
    """
    Compute the precision precision@k normalized by group.
    :param eval_true: Evaluation labels.
    :param eval_pred: Evaluation predictions.
    :param k: k value.
    :return: Precision@k
    """
    
    if k > len(eval_pred):
        k = len(eval_pred)
    
    eval_top_index = torch.topk(eval_pred, k, sorted=False).indices.cpu()
    
    eval_tp = eval_true[eval_top_index].sum().item()
    pk = eval_tp / k

    return pk


def precision_at_k(eval_true, eval_pred, k):
    """
    Compute precision@k.
    :param eval_true: Evaluation labels.
    :param eval_pred: Evaluation predictions.
    :param k: k value.
    :return: Precision@k
    """
    
    if k > len(eval_pred):
        k = len(eval_pred)
    
    eval_top_index = torch.topk(eval_pred, k, sorted=False).indices.cpu()
    
    eval_tp = eval_true[eval_top_index].sum().item()
    pk = eval_tp / k



    return pk
    
def hits_at_k(eval_true, eval_pred, k):
    """
    Compute hits@k.

    :param eval_true: Evaluation labels.
    :param eval_pred: Evaluation predictions.
    :param k: k value.
    :return: Hits@k.
    """

    pred_pos = eval_pred[eval_true == 1]
    pred_neg = eval_pred[eval_true == 0]
    try:
        kth_score_in_negative_edges = torch.topk(pred_neg, k)[0][-1]
    except:
        code.interact(local=locals())
    hitsk = float(torch.sum(pred_pos > kth_score_in_negative_edges).cpu()) / len(pred_pos)
    return hitsk


def average_precision(eval_true, eval_pred):
    """
    Compute Average Precision (AP).

    :param eval_true: Evaluation labels.
    :param eval_pred: Evaluation predictions.
    :return: AP.
    """

    return average_precision_score(eval_true, eval_pred.cpu())


def mean_reciprocal_rank(eval_true, eval_pred):

    pred_pos = eval_pred[eval_true == 1]
    pred_neg = eval_pred[eval_true == 0]
    pred = torch.cat([pred_pos, pred_neg])
    sorted_index = pred.argsort(descending=True)
    ranks_including_pos = torch.argwhere(sorted_index < len(pred_pos)) + 1
    ranks = ranks_including_pos.flatten() - torch.arange(0, len(pred_pos)).to(ranks_including_pos.device)

    return (1 / ranks).mean().item()


def auc(target, preds):
    return binary_auroc(preds, target).item()


def normalized_precision_at_k(eval_true, eval_pred, eval_groups, k):
    """
    Compute the precision precision@k normalized by group.
    :param eval_true: Evaluation labels.
    :param eval_pred: Evaluation predictions.
    :param k: k value.
    :return: Precision@k
    """
    
    if k > len(eval_pred):
        k = len(eval_pred)
    
    eval_top_index = torch.topk(eval_pred, k, sorted=False).indices.cpu()
    
    eval_intra_group = (eval_groups[eval_top_index] == 0).sum()
    eval_inter_group = (eval_groups[eval_top_index] == 1).sum()
    
    total_intra_group = torch.sum(eval_true == 0)
    total_inter_group = torch.sum(eval_true == 1)
    
    
    p_eval_intra_group = eval_intra_group / total_intra_group
    p_eval_inter_group = eval_inter_group / total_inter_group

    return p_eval_intra_group, p_eval_inter_group
