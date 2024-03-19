"""
hitrate@k, mean reciprocal rank (MRR) and Area under the receiver operator characteristic curve (AUC) evaluation metrics
"""
from sklearn.metrics import roc_auc_score
import torch

def evaluate_hits(evaluator, pos_train_pred, neg_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred,
                  Ks=[20, 50, 100], use_val_negs_for_train=True):
    """
    Evaluate the hit rate at K
    :param evaluator: an ogb Evaluator object
    :param pos_val_pred: Tensor[val edges]
    :param neg_val_pred: Tensor[neg val edges]
    :param pos_test_pred: Tensor[test edges]
    :param neg_test_pred: Tensor[neg test edges]
    :param Ks: top ks to evaluatate for
    :return: dic[ks]
    """
    results = {}
    # As the training performance is used to assess overfitting it can help to use the same set of negs for
    # train and val comparisons.
    if use_val_negs_for_train:
        neg_train = neg_val_pred
    else:
        neg_train = neg_train_pred
    for K in Ks:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_train,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_val_pred,
            'y_pred_neg': neg_val_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results


def evaluate_mrr(evaluator, pos_train_pred, neg_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    """
    Evaluate the mean reciprocal rank at K
    :param evaluator: an ogb Evaluator object
    :param pos_val_pred: Tensor[val edges]
    :param neg_val_pred: Tensor[neg val edges]
    :param pos_test_pred: Tensor[test edges]
    :param neg_test_pred: Tensor[neg test edges]
    :param Ks: top ks to evaluatate for
    :return: dic with single key 'MRR'
    """
    neg_train_pred = neg_train_pred.view(pos_train_pred.shape[0], -1)
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    results = {}

    train_mrr = evaluator.eval({
        'y_pred_pos': pos_train_pred,
        # for mrr negs all have the same src, so can't use the val negs, but as long as the same  number of negs / pos are
        # used the results will be comparable.
        'y_pred_neg': neg_train_pred,
    })['mrr_list'].mean().item()

    valid_mrr = evaluator.eval({
        'y_pred_pos': pos_val_pred,
        'y_pred_neg': neg_val_pred,
    })['mrr_list'].mean().item()

    test_mrr = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })['mrr_list'].mean().item()

    results['MRR'] = (train_mrr, valid_mrr, test_mrr)

    return results


def evaluate_auc(val_pred, val_true, test_pred, test_true):
    """
    the ROC AUC
    :param val_pred: Tensor[val edges] predictions
    :param val_true: Tensor[val edges] labels
    :param test_pred: Tensor[test edges] predictions
    :param test_true: Tensor[test edges] labels
    :return:
    """
    valid_auc = roc_auc_score(val_true, val_pred)
    test_auc = roc_auc_score(test_true, test_pred)
    results = {}
    results['AUC'] = (valid_auc, test_auc)

    return results


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



def positive_rate_disparity(eval_pred, eval_groups):
    """
    Compute the positive disparity.
    :param eval_true: Evaluation labels.
    :param eval_pred: Evaluation predictions.
    :return: Positive disparity.
    """
    eval_pred = torch.sigmoid(eval_pred)
    
    intra_group = (eval_groups == 0) | (eval_groups == 2)
    inter_group = (eval_groups == 1)
    
    probability_of_a_link_given_group = lambda preds, group_size: torch.sum(preds * (1 / group_size))
    
    p_eval_intra_group, p_eval_inter_group = probability_of_a_link_given_group(eval_pred[intra_group], intra_group.sum()), probability_of_a_link_given_group(eval_pred[inter_group], inter_group.sum())
    return p_eval_inter_group - p_eval_intra_group


def true_positive_rate_disparity(eval_true, eval_pred, eval_groups):
    """
    Compute the positive disparity.
    :param eval_true: Evaluation labels.
    :param eval_pred: Evaluation predictions.
    :return: Positive disparity.
    """
    # import code
    # code.interact(local={**locals(), **globals()})
    eval_pred = torch.sigmoid(eval_pred)
    
    intra_group = ((eval_groups == 0) | (eval_groups == 2)) & (eval_true == 1)
    inter_group = (eval_groups == 1) & (eval_true == 1)
    
    probability_of_a_link_given_group = lambda preds, group_size: torch.sum(preds * (1 / group_size))

    p_eval_intra_group, p_eval_inter_group = probability_of_a_link_given_group(eval_pred[intra_group], intra_group.sum()), probability_of_a_link_given_group(eval_pred[inter_group], inter_group.sum())
    return p_eval_inter_group - p_eval_intra_group