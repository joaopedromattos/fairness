import argparse
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
from torch.utils.tensorboard import SummaryWriter
from utils import PermIterator
import time
from ogbdataset import loaddataset
from typing import Iterable
from fairness import FairLearner

import wandb

from evaluation import true_positive_rate_disparity
import datetime

import os

from torchmetrics.functional.classification import binary_accuracy

torch.autograd.set_detect_anomaly(True)


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
          alpha: float=None):
    
    if alpha is not None:
        predictor.setalpha(alpha)
    
    model.train()
    fair_model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    pos_train_edge = pos_train_edge.t()

    total_loss = []
    total_fair_loss = []
    adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)
            
    negedge = split_edge['train']['edge_neg'].to(data.x.device).t()
    
    for perm in PermIterator(
            adjmask.device, adjmask.shape[0], min(adjmask.shape[0], batch_size)
    ):
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
        edge = pos_train_edge[:, perm]
        
        if not link_level:
            pos_outs = predictor.multidomainforward(h,
                                                    adj,
                                                    edge,
                                                    cndropprobs=cnprobs)

            pos_losss = -F.logsigmoid(pos_outs).mean()
            edge = negedge[:, perm]
            neg_outs = predictor.multidomainforward(h, adj, edge, cndropprobs=cnprobs)
            neg_losss = -F.logsigmoid(-neg_outs).mean()
        else:
            pos_outs, pos_embs = predictor.multidomainforward(h,
                                                    adj,
                                                    edge,
                                                    cndropprobs=cnprobs)

            pos_losss = -F.logsigmoid(pos_outs).mean()
            edge = negedge[:, perm]
            neg_outs, neg_embs = predictor.multidomainforward(h, adj, edge, cndropprobs=cnprobs)
            neg_losss = -F.logsigmoid(-neg_outs).mean()
            
        if not no_intervention:
            if link_level:
                protected_groups_labels_pos = (data.y[pos_train_edge[:, perm]].sum(0).long() == 1).float()
                protected_groups_labels_neg = (data.y[negedge[:, perm]].sum(0).long() == 1).float()
                
                pos_fair_outs = fair_model(pos_embs)
                neg_fair_outs = fair_model(neg_embs)
                
                fairness_loss = F.binary_cross_entropy_with_logits(pos_fair_outs.squeeze(-1), protected_groups_labels_pos) + F.binary_cross_entropy_with_logits(neg_fair_outs.squeeze(-1), protected_groups_labels_neg)
                
                fairness_loss.backward(retain_graph=True)
                print("Gradient sum: fair_model.lins[1].weight.grad.sum()", fair_model.lins[1].weight.grad.sum(), flush=True)
                
                loss = neg_losss + pos_losss - (reg_lambda * fairness_loss)
                total_fair_loss.append(fairness_loss.item())
                
                loss.backward()
                fair_optimizer.step()
                optimizer.step()
                
            else:
                fair_pred = fair_model(h)
                
                fairness_loss = F.binary_cross_entropy_with_logits(fair_pred.squeeze(-1), data.y.float())
                
                fairness_loss.backward(retain_graph=True)

                print("Gradient sum: fair_model.lins[1].weight.grad.sum()", fair_model.lins[1].weight.grad.sum(), flush=True)
                                            
                loss = neg_losss + pos_losss - (reg_lambda * fairness_loss)
                total_fair_loss.append(fairness_loss.item())
                
                loss.backward()
                fair_optimizer.step()
                optimizer.step()
                
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
        
        fair_pred = fair_model(h)
        fairness_acc = binary_accuracy(torch.sigmoid(fair_pred.view(-1)).cpu(), data.y.float().cpu())
        fairness_train, fairness_test = fairness_acc, fairness_acc
        
    else:

        pos_train_pred = torch.empty(0)
        pos_train_fair_pred = torch.empty(0)
        for perm in PermIterator(pos_train_edge.device, pos_train_edge.shape[0], batch_size, False):
            pred, pred_embs = predictor(h, adj, pos_train_edge[perm].t())
            pos_train_pred = torch.cat((pos_train_pred.squeeze().cpu(), pred.view(-1).cpu()), dim=0)
            pos_train_fair_pred = torch.cat([pos_train_fair_pred.cpu(), fair_model(pred_embs).cpu()], dim=0)

        neg_train_pred = torch.empty(0)
        neg_train_fair_pred = torch.empty(0)
        for perm in PermIterator(neg_train_edge.device, neg_train_edge.shape[0], batch_size, False):
            pred, pred_embs = predictor(h, adj, neg_train_edge[perm].t())
            neg_train_pred = torch.cat((neg_train_pred.squeeze().cpu(), pred.view(-1).cpu()), dim=0)
            neg_train_fair_pred = torch.cat([neg_train_fair_pred.cpu(), fair_model(pred_embs).cpu()], dim=0)

        pos_valid_pred = torch.empty(0)
        pos_valid_fair_pred = torch.empty(0)
        for perm in PermIterator(pos_valid_edge.device, pos_valid_edge.shape[0], batch_size, False):
            pred, pred_embs = predictor(h, adj, pos_valid_edge[perm].t())
            pos_valid_pred = torch.cat((pos_valid_pred.squeeze().cpu(), pred.view(-1).cpu()), dim=0)
            pos_valid_fair_pred = torch.cat([pos_valid_fair_pred.cpu(), fair_model(pred_embs).cpu()], dim=0)

        neg_valid_pred = torch.empty(0)
        neg_valid_fair_pred = torch.empty(0)
        for perm in PermIterator(neg_valid_edge.device, neg_valid_edge.shape[0], batch_size, False):
            pred, pred_embs = predictor(h, adj, neg_valid_edge[perm].t())
            neg_valid_pred = torch.cat((neg_valid_pred.squeeze().cpu(), pred.view(-1).cpu()), dim=0)
            neg_valid_fair_pred = torch.cat([neg_valid_fair_pred.cpu(), fair_model(pred_embs).cpu()], dim=0)

        if use_valedges_as_input:
            adj = data.full_adj_t
            h = model(data.x, adj)

        pos_test_pred = torch.empty(0)
        pos_test_fair_pred = torch.empty(0)
        for perm in PermIterator(pos_test_edge.device, pos_test_edge.shape[0], batch_size, False):
            pred, pred_embs = predictor(h, adj, pos_test_edge[perm].t())
            pos_test_pred = torch.cat((pos_test_pred.squeeze().cpu(), pred.view(-1).cpu()), dim=0)
            pos_test_fair_pred = torch.cat([pos_test_fair_pred.cpu(), fair_model(pred_embs).cpu()], dim=0)

        neg_test_pred = torch.empty(0)
        neg_test_fair_pred = torch.empty(0)
        for perm in PermIterator(neg_test_edge.device, neg_test_edge.shape[0], batch_size, False):
            pred, pred_embs = predictor(h, adj, neg_test_edge[perm].t())
            neg_test_pred = torch.cat((neg_test_pred.squeeze().cpu(), pred.view(-1).cpu()), dim=0)
            neg_test_fair_pred = torch.cat([neg_test_fair_pred.cpu(), fair_model(pred_embs).cpu()], dim=0)
            
            
        fair_train_preds, fair_train_labels = torch.cat([pos_train_fair_pred, neg_train_fair_pred]), torch.cat([torch.ones_like(pos_train_fair_pred), torch.zeros_like(neg_train_fair_pred)])
        fair_valid_preds, fair_valid_labels = torch.cat([pos_valid_fair_pred, neg_valid_fair_pred]), torch.cat([torch.ones_like(pos_valid_fair_pred), torch.zeros_like(neg_valid_fair_pred)])
        fair_test_preds, fair_test_labels = torch.cat([pos_test_fair_pred, neg_test_fair_pred]), torch.cat([torch.ones_like(pos_test_fair_pred), torch.zeros_like(neg_test_fair_pred)])
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
    
    train_tprd = true_positive_rate_disparity(train_labels, train_preds, train_protected_groups_labels.detach()).abs()
    val_tprd = true_positive_rate_disparity(val_labels, val_preds, val_protected_groups_labels.detach()).abs()
    test_tprd = true_positive_rate_disparity(test_labels, test_preds, test_protected_groups_labels.detach()).abs()
    
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

        results[f'rep{0}_TrainHits@{K}'] = train_hits
        results[f'rep{0}_ValHits@{K}'] = valid_hits
        results[f'rep{0}_TestHits@{K}'] = test_hits
        
    results[f'rep{0}_true_positive_rate_disparity_Train'] = train_tprd.item()
    results[f'rep{0}_true_positive_rate_disparity_Valid'] = val_tprd.item()
    results[f'rep{0}_true_positive_rate_disparity_Test'] = test_tprd.item()
    results[f'rep{0}_adv_acc_train'] = fairness_train.item()
    results[f'rep{0}_adv_acc'] = fairness_test.item()
    
    return results, h.cpu(), saved_output


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_valedges_as_input', action='store_true', help="whether to add validation edges to the input adjacency matrix of gnn")
    parser.add_argument('--epochs', type=int, default=40, help="number of epochs")
    parser.add_argument('--runs', type=int, default=3, help="number of repeated runs")
    parser.add_argument('--device', type=int, default=0, help="which device to use to run the process")
    parser.add_argument('--dataset', type=str, default="collab")
    
    parser.add_argument('--batch_size', type=int, default=8192, help="batch size")
    parser.add_argument('--testbs', type=int, default=8192, help="batch size for test")
    parser.add_argument('--maskinput', action="store_true", help="whether to use target link removal")

    parser.add_argument('--mplayers', type=int, default=1, help="number of message passing layers")
    parser.add_argument('--nnlayers', type=int, default=3, help="number of mlp layers")
    parser.add_argument('--hiddim', type=int, default=32, help="hidden dimension")
    parser.add_argument('--ln', action="store_true", help="whether to use layernorm in MPNN")
    parser.add_argument('--lnnn', action="store_true", help="whether to use layernorm in mlp")
    parser.add_argument('--res', action="store_true", help="whether to use residual connection")
    parser.add_argument('--jk', action="store_true", help="whether to use JumpingKnowledge connection")
    parser.add_argument('--gnndp', type=float, default=0.3, help="dropout ratio of gnn")
    parser.add_argument('--xdp', type=float, default=0.3, help="dropout ratio of gnn")
    parser.add_argument('--tdp', type=float, default=0.3, help="dropout ratio of gnn")
    parser.add_argument('--gnnedp', type=float, default=0.3, help="edge dropout ratio of gnn")
    parser.add_argument('--predp', type=float, default=0.3, help="dropout ratio of predictor")
    parser.add_argument('--preedp', type=float, default=0.3, help="edge dropout ratio of predictor")
    parser.add_argument('--gnnlr', type=float, default=0.0003, help="learning rate of gnn")
    parser.add_argument('--prelr', type=float, default=0.0003, help="learning rate of predictor")
    
    parser.add_argument('--adv_lr', type=float, default=0.0001, help="learning rate of fairness model")
    parser.add_argument('--reg_lambda', type=float, default=0.0001, help="learning rate of fairness model")
    
    # detailed hyperparameters
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument("--use_xlin", action="store_true")
    parser.add_argument("--tailact", action="store_true")
    parser.add_argument("--twolayerlin", action="store_true")
    parser.add_argument("--increasealpha", action="store_true")
    
    parser.add_argument('--splitsize', type=int, default=-1, help="split some operations inner the model. Only speed and GPU memory consumption are affected.")

    # parameters used to calibrate the edge existence probability in NCNC
    parser.add_argument('--probscale', type=float, default=5)
    parser.add_argument('--proboffset', type=float, default=3)
    parser.add_argument('--pt', type=float, default=0.5)
    parser.add_argument("--learnpt", action="store_true")

    # For scalability, NCNC samples neighbors to complete common neighbor. 
    parser.add_argument('--trndeg', type=int, default=-1, help="maximum number of sampled neighbors during the training process. -1 means no sample")
    parser.add_argument('--tstdeg', type=int, default=-1, help="maximum number of sampled neighbors during the test process")
    # NCN can sample common neighbors for scalability. Generally not used. 
    parser.add_argument('--cndeg', type=int, default=-1)
    
    # predictor used, such as NCN, NCNC
    parser.add_argument('--predictor', choices=predictor_dict.keys())
    parser.add_argument("--depth", type=int, default=1, help="number of completion steps in NCNC")
    # gnn used, such as gin, gcn.
    parser.add_argument('--model', choices=convdict.keys())

    parser.add_argument('--save_gemb', action="store_true", help="whether to save node representations produced by GNN")
    parser.add_argument('--load', type=str, help="where to load node representations produced by GNN")
    parser.add_argument("--loadmod", action="store_true", help="whether to load trained models")
    parser.add_argument("--savemod", action="store_true", help="whether to save trained models")
    
    parser.add_argument("--savex", action="store_true", help="whether to save trained node embeddings")
    parser.add_argument("--loadx", action="store_true", help="whether to load trained node embeddings")
    
    parser.add_argument("--no_intervention", action="store_true", help="removes the intervention model")
    
    parser.add_argument("--link_level", action="store_true", help="link level fair model prediction")
    
    parser.add_argument("--node_split", action="store_true", help="wandb sweep")
    
    parser.add_argument("--no_wandb", action="store_true", help="no wandb")
    parser.add_argument("--wandb_sweep", action="store_true", help="wandb sweep")
   
    # not used in experiments
    parser.add_argument('--cnprob', type=float, default=0)
    args = parser.parse_args()
    return args


def main():
    print(args, flush=True)
    
    
    dataset_path = {
        'facebook': '/home/jrm28/fairness/subgraph_sketching-original/dataset/ego-facebook/processed/facebook_1684.pt',
        'facebook_graphair': "/home/jrm28/fairness/graphair/fairgraph/method/checkpoint/out/AUGMENTED_facebook_10000_epochs_2024-03-13_14-50-37/splits.pt",
        'gplus': '/home/jrm28/fairness/subgraph_sketching-original/dataset/gplus/processed/gplus_100129275726588145876.pt',
        'sbm': '/home/jrm28/fairness/subgraph_sketching-original/dataset/sbm/processed/sbm.pt',
        'sbm_medium': '/home/jrm28/fairness/subgraph_sketching-original/dataset/sbm/processed/sbm_medium.pt',
        'sbm_bigger': '/home/jrm28/fairness/subgraph_sketching-original/dataset/sbm/processed/sbm_bigger.pt',
    }[args.dataset]

    hpstr = str(args).replace(" ", "").replace("Namespace(", "").replace(
        ")", "").replace("True", "1").replace("False", "0").replace("=", "").replace("epochs", "").replace("runs", "").replace("save_gemb", "")
    writer = SummaryWriter(f"./rec/{args.model}_{args.predictor}")
    writer.add_text("hyperparams", hpstr)

    evaluator = Evaluator(name=f'ogbl-ppa')

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    # device = torch.cuda.set_device(0)
    # device='cpu'
    
    dataset_file = f"dataset/splits/{args.dataset}{'_node_split' if args.node_split else ''}.pt" 
    if os.path.isfile(dataset_file):
        data, split_edge = torch.load(dataset_file)
    else:
        data, split_edge = loaddataset(args.dataset, dataset_path, args.use_valedges_as_input, args.load, args)
        torch.save((data, split_edge), dataset_file)
    
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
        # run_name += f'{"NCN" if args.predictor == "cn1" else "NCNC"}_{args.dataset}'
        if args.predictor == "cn1":
            run_name += f'{"NCN"}_{args.dataset}'
        elif args.predictor == "cn0":
            run_name += f'{"GAE"}_{args.dataset}'
        else:
            run_name += f'{"NCNC"}_{args.dataset}'
            
        run_name += "_link_level" if args.link_level else ""
        run_name += "_node_split" if args.node_split else ""
        wandb_run = wandb.init(project="lpfairness", entity="joaopedromattos", config=args, name=run_name, mode="online" if not args.no_wandb else "disabled")

        artifact = wandb.Artifact(args.dataset, type="dataset")
        artifact.add_reference(f"file:///{dataset_path}")
        
        wandb_run.use_artifact(artifact)
        
        if args.wandb_sweep:
            args.adv_lr = wandb.config.adv_lr
            args.reg_lambda = wandb.config.reg_lambda

        
        set_seed(run)
        # if args.dataset in ["Cora", "Citeseer", "Pubmed", "facebook", "gplus", "sbm", "sbm_medium", "sbm_bigger"]:
        #     data, split_edge = loaddataset(args.dataset, args.use_valedges_as_input, args.load) # get a new split of dataset
        #     data = data.to(device)
        bestscore = None
        
        # build model
        model = GCN(data.num_features, args.hiddim, args.hiddim, args.mplayers,
                    args.gnndp, args.ln, args.res, data.max_x,
                    args.model, args.jk, args.gnnedp,  xdropout=args.xdp, taildropout=args.tdp, noinputlin=args.loadx).to(device)
        
        fair_model = FairLearner(args.hiddim, args.hiddim, 2).to(device)
        
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
        for epoch in range(1, 1 + args.epochs):
            alpha = max(0, min((epoch-5)*0.1, 1)) if args.increasealpha else None
            t1 = time.time()
            print("training")
            loss, fair_loss = train(model, fair_model, predictor, data, split_edge, optimizer, fair_optimizer,
                         args.batch_size, args.reg_lambda, args.no_intervention, args.link_level, args.maskinput, [], alpha)
            print("after training")
            print(f"trn time {time.time()-t1:.2f} s", flush=True)
            if True:
                t1 = time.time()
                results, h, saved_output = test(model, fair_model, predictor, data, split_edge, evaluator,
                               args.testbs, args.link_level, args.use_valedges_as_input)
                print(f"test time {time.time()-t1:.2f} s")
                
                print(results, flush=True)
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                torch.save(saved_output, f"saved_output/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}_{epoch}{timestamp}.pt")

                if True:
                    
                    if bestscore is None or bestscore[f"rep{0}_ValHits@{100}"] < results[f"rep{0}_ValHits@{100}"]:
                        train_hits, valid_hits, test_hits = results[f'rep{0}_TrainHits@{100}'], results[f'rep{0}_ValHits@{100}'], results[f'rep{0}_TestHits@{100}']
                        if args.save_gemb:
                            torch.save(h, f"gemb/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}.pt")
                        if args.savex:
                            torch.save(model.xemb[0].weight.detach(), f"gemb/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pt")
                        if args.savemod:
                            torch.save(model.state_dict(), f"gmodel/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pt")
                            torch.save(predictor.state_dict(), f"gmodel/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pre.pt")
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
                "rep0_true_positive_rate_disparity_Train": results["rep0_true_positive_rate_disparity_Train"],
                "rep0_true_positive_rate_disparity_Valid": results["rep0_true_positive_rate_disparity_Valid"],
                "rep0_true_positive_rate_disparity_Test": results["rep0_true_positive_rate_disparity_Test"],
                f'rep{0}_adv_acc_train': results[f'rep{0}_adv_acc_train'],
                f'rep{0}_adv_acc': results[f'rep{0}_adv_acc'],
                "epoch_step" : epoch - 1
            })
            
        ret.append(bestscore)
        
        wandb.finish()
            
            
    #     print(f"best {bestscore}")
    #     if args.dataset == "collab":
    #         ret.append(bestscore["Hits@50"][-2:])
    #     elif args.dataset == "ppa":
    #         ret.append(bestscore["Hits@100"][-2:])
    #     elif args.dataset == "ddi":
    #         ret.append(bestscore["Hits@20"][-2:])
    #     elif args.dataset == "citation2":
    #         ret.append(bestscore[-2:])
    #     elif args.dataset in ["Pubmed", "Cora", "Citeseer"]:
    #         ret.append(bestscore["Hits@100"][-2:])
    #     else:
    #         raise NotImplementedError
    # ret = np.array(ret)
    # print(ret)
    # print(f"Final result: val {np.average(ret[:, 0]):.4f} {np.std(ret[:, 0]):.4f} tst {np.average(ret[:, 1]):.4f} {np.std(ret[:, 1]):.4f}")


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
            },
        }

        sweep_id = wandb.sweep(sweep_configuration, project="lpfairness", entity="joaopedromattos")
        
        wandb.agent(sweep_id, function=main)
    else:
        main()
