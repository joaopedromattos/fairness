"""
main module
"""
import argparse
import time
import warnings
from math import inf
import sys

sys.path.insert(0, '..')

import numpy as np
import torch
from ogb.linkproppred import Evaluator

torch.set_printoptions(precision=4)
import wandb
# when generating subgraphs the supervision edge is deleted, which triggers a SparseEfficiencyWarning, but this is
# not a performance bottleneck, so suppress for now
from scipy.sparse import SparseEfficiencyWarning

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

from src.data import get_data, get_loaders
from src.models.adversary import AdversaryLearner
from src.models.elph import ELPH, BUDDY
from src.models.seal import SEALDGCNN, SEALGCN, SEALGIN, SEALSAGE
from src.utils import ROOT_DIR, print_model_params, select_embedding, str2bool
from src.wandb_setup import initialise_wandb
from src.runners.train import get_train_func
from src.runners.inference import test
from types import SimpleNamespace

from torchmetrics.functional.classification import binary_accuracy
import torch_geometric



def run():
    
    dataset_path = {
        'facebook': '/home/jrm28/fairness/subgraph_sketching-original/dataset/ego-facebook/processed/facebook_1684.pt',
        'gplus': '/home/jrm28/fairness/subgraph_sketching-original/dataset/gplus/processed/gplus_100129275726588145876.pt',
    }[args.dataset_name]
    
    # Create a timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    results_dir = f'/home/jrm28/fairness/subgraph_sketching-original/src/results/{args.dataset_name}_{timestamp}/'
    preds_dir = results_dir + f'/preds/'
    models_dir = results_dir + f'/model/'

    torch_geometric.data.makedirs(results_dir)
    torch_geometric.data.makedirs(preds_dir)
    torch_geometric.data.makedirs(models_dir)

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f"executing on {device}")
    results_list = []
    train_func = get_train_func(args)
    for rep in range(args.reps):
        
        torch.manual_seed(rep)
        
        wandb_run = wandb.init(project="lpfairness", entity="joaopedromattos", config=args, name=f'BUDDY_{args.dataset_name}', mode="online")
    
        artifact = wandb.Artifact(args.dataset_name, type="dataset")
        artifact.add_reference(f"file:///{dataset_path}")
        
        wandb_run.use_artifact(artifact)

        # getting data from W&B Sweeps
        if args.wandb_sweep:
            args.adv_lr = wandb.config.adv_lr
            args.reg_lambda = wandb.config.reg_lambda
        
        dataset, splits, directed, eval_metric = get_data(args, dataset_path)
        
        data = dataset
        dataset = SimpleNamespace(data=dataset)
        
        
        train_loader, train_eval_loader, val_loader, test_loader = get_loaders(args, dataset, splits, directed)
        # if args.dataset_name.startswith('ogbl'):  # then this is one of the ogb link prediction datasets
        #     evaluator = Evaluator(name=args.dataset_name)
        # else:
        #     evaluator = Evaluator(name='ogbl-ppa')  # this sets HR@100 as the metric
        evaluator = Evaluator(name='ogbl-ppa')
        emb = select_embedding(args, data.num_nodes, device)
        model, adv_model, optimizer, adv_optimizer = select_model(args, data, emb, device)
        
        val_res = test_res = best_epoch = 0
        print(f'running repetition {rep}')
        if rep == 0:
            print_model_params(model)
        
        for epoch in range(args.epochs):
            t0 = time.time()

            loss, adv_loss, lp_loss = train_func(model, adv_model, optimizer, adv_optimizer, train_loader, args, device)
            if ((epoch + 1) % args.eval_steps == 0) or (epoch == args.epochs - 1):
                results, test_pred, test_true, test_adv_logits, test_adv_labels, test_groups, fairness_results = test(model, adv_model, evaluator, train_eval_loader, val_loader, test_loader, args, device,
                               eval_metric=eval_metric)

                for key, result in results.items():
                    # import code
                    # code.interact(local={**globals(), **locals()})
                    train_res, tmp_val_res, tmp_test_res = result
                    if tmp_val_res > val_res:
                        val_res = tmp_val_res
                        test_res = tmp_test_res
                        best_epoch = epoch
                    res_dic = {f'rep{0}_loss': loss, f'rep{0}_Train' + key: 100 * train_res,
                               f'rep{0}_adv_loss': adv_loss, f'rep{0}_Train' + key: 100 * train_res,
                               f'rep{0}_lp_loss': lp_loss, f'rep{0}_Train' + key: 100 * train_res,
                               f'rep{0}_Val' + key: 100 * val_res, f'rep{0}_tmp_val' + key: 100 * tmp_val_res,
                               f'rep{0}_tmp_test' + key: 100 * tmp_test_res,
                               f'rep{0}_Test' + key: 100 * test_res, f'rep{0}_best_epoch': best_epoch,
                               f'rep{0}_epoch_time': time.time() - t0, 'epoch_step': epoch,
                               f'rep{0}_normalized_prec@100_intra_Train': fairness_results['train']['precision_intra'],
                               f'rep{0}_normalized_prec@100_inter_Train': fairness_results['train']['precision_inter'],
                               f'rep{0}_normalized_prec@100_intra_Test': fairness_results['test']['precision_intra'],
                               f'rep{0}_normalized_prec@100_inter_Test': fairness_results['test']['precision_inter'],
                               f'rep{0}_positive_rate_disparity_Train': fairness_results['train']['positive_rate_disparity'].abs(),
                               f'rep{0}_positive_rate_disparity_Val': fairness_results['val']['positive_rate_disparity'].abs(),
                               f'rep{0}_positive_rate_disparity_Test': fairness_results['test']['positive_rate_disparity'].abs(),
                               f'rep{0}_true_positive_rate_disparity_Train': fairness_results['train']['true_positive_rate_disparity'].abs(),
                               f'rep{0}_true_positive_rate_disparity_Val': fairness_results['val']['true_positive_rate_disparity'].abs(),
                               f'rep{0}_true_positive_rate_disparity_Test': fairness_results['test']['true_positive_rate_disparity'].abs(),
                               f'rep{0}_adv_acc': binary_accuracy(torch.sigmoid(test_adv_logits).cpu(), test_adv_labels)}
                    if args.wandb:
                        wandb.log(res_dic)
                        print("log_wandb")
                    to_print = f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * train_res:.2f}%, Valid: ' \
                               f'{100 * val_res:.2f}%, Test: {100 * test_res:.2f}%, epoch time: {time.time() - t0:.1f}'
                    print(key)
                    print(to_print)

                torch.save({"test_pred" : test_pred, "test_true" : test_true}, f'{preds_dir}/{args.dataset_name}_{rep}_{epoch}.pth')
                torch.save({"test_pred_adv" : test_adv_logits, "test_true_adv" : test_adv_labels, "test_groups" : test_groups}, f'{preds_dir}/{args.dataset_name}_{rep}_{epoch}_adv.pth')
                
                # save optimizer
                torch.save(optimizer.state_dict(), models_dir + f'{args.dataset_name}_{rep}_{epoch}_optimizer.pth')


        results_list.append([test_res, val_res, train_res])

        test_acc_mean, val_acc_mean, train_acc_mean = np.mean(results_list, axis=0) * 100
        test_acc_std = np.sqrt(np.var(results_list, axis=0)[0]) * 100
        wandb_results = {'test_mean': test_acc_mean, 'val_mean': val_acc_mean, 'train_mean': train_acc_mean,
                            'test_acc_std': test_acc_std}

        print(wandb_results)


        if args.save_model:
            
            torch.save(model.state_dict(), models_dir + f'{args.dataset_name}_{rep}_{epoch}.pth')
            
            model_params_artifact = wandb.Artifact(f"buddy_{timestamp}", type="weights")
            model_params_artifact.add_reference(f"file://{models_dir + f'/{args.dataset_name}_{rep}_{epoch}.pth'}")
            wandb_run.use_artifact(model_params_artifact)
            
            preds_params_artifact = wandb.Artifact(f"buddy_preds_{timestamp}", type="scores")
            preds_params_artifact.add_reference(f"file://{preds_dir}")
            wandb_run.use_artifact(preds_params_artifact)

        wandb.finish()


def select_model(args, dataset, emb, device):
    if args.model == 'SEALDGCNN':
        model = SEALDGCNN(args.hidden_channels, args.num_seal_layers, args.max_z, args.sortpool_k,
                          dataset, args.dynamic_train, use_feature=args.use_feature,
                          node_embedding=emb).to(device)
    elif args.model == 'SEALSAGE':
        model = SEALSAGE(args.hidden_channels, args.num_seal_layers, args.max_z, dataset.num_features,
                         args.use_feature, node_embedding=emb, dropout=args.dropout).to(device)
    elif args.model == 'SEALGCN':
        model = SEALGCN(args.hidden_channels, args.num_seal_layers, args.max_z, dataset.num_features,
                        args.use_feature, node_embedding=emb, dropout=args.dropout, pooling=args.seal_pooling).to(
            device)
    elif args.model == 'SEALGIN':
        model = SEALGIN(args.hidden_channels, args.num_seal_layers, args.max_z, dataset.num_features,
                        args.use_feature, node_embedding=emb, dropout=args.dropout).to(device)
    elif args.model == 'BUDDY':
        model = BUDDY(args, dataset.num_features, node_embedding=emb).to(device)
    elif args.model == 'ELPH':
        model = ELPH(args, dataset.num_features, node_embedding=emb).to(device)
    else:
        raise NotImplementedError
    parameters = list(model.parameters())
    if args.train_node_embedding:
        torch.nn.init.xavier_uniform_(emb.weight)
        parameters += list(emb.parameters())
    optimizer = torch.optim.Adam(params=parameters, lr=args.lr, weight_decay=args.weight_decay)
    total_params = sum(p.numel() for param in parameters for p in param)
    print(f'Total number of parameters is {total_params}')
    if args.model == 'DGCNN':
        print(f'SortPooling k is set to {model.k}')
        
    adv_model = AdversaryLearner(input_channels=model.lin.in_features, hidden_channels=args.hidden_channels, hidden_layers=8).to(device)
    adv_optimizer = torch.optim.Adam(params=adv_model.parameters(), lr=args.adv_lr)
    
    return model, adv_model, optimizer, adv_optimizer


if __name__ == '__main__':
    # Data settings
    parser = argparse.ArgumentParser(description='Efficient Link Prediction with Hashes (ELPH)')
    parser.add_argument('--dataset_name', type=str, default='Cora',
                        choices=['Cora', 'CiteSeer', 'PubMed', 'ogbl-ppa', 'ogbl-collab', 'ogbl-ddi',
                                 'ogbl-citation2', 'facebook', 'gplus'])
    parser.add_argument('--val_pct', type=float, default=0.1,
                        help='the percentage of supervision edges to be used for validation. These edges will not appear'
                             ' in the training set and will only be used as message passing edges in the test set')
    parser.add_argument('--test_pct', type=float, default=0.2,
                        help='the percentage of supervision edges to be used for test. These edges will not appear'
                             ' in the training or validation sets for either supervision or message passing')
    parser.add_argument('--train_samples', type=float, default=inf, help='the number of training edges or % if < 1')
    parser.add_argument('--val_samples', type=float, default=inf, help='the number of val edges or % if < 1')
    parser.add_argument('--test_samples', type=float, default=inf, help='the number of test edges or % if < 1')
    parser.add_argument('--preprocessing', type=str, default=None)
    parser.add_argument('--sign_k', type=int, default=0)
    parser.add_argument('--load_features', action='store_true', help='load node features from disk')
    parser.add_argument('--load_hashes', action='store_true', help='load hashes from disk')
    parser.add_argument('--cache_subgraph_features', action='store_true',
                        help='write / read subgraph features from disk')
    parser.add_argument('--train_cache_size', type=int, default=inf, help='the number of training edges to cache')
    parser.add_argument('--year', type=int, default=0, help='filter training data from before this year')
    # GNN settings
    parser.add_argument('--model', type=str, default='BUDDY')
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--eval_batch_size', type=int, default=1000000,
                        help='eval batch size should be largest the GPU memory can take - the same is not necessarily true at training time')
    parser.add_argument('--label_dropout', type=float, default=0.5)
    parser.add_argument('--feature_dropout', type=float, default=0.5)
    parser.add_argument('--sign_dropout', type=float, default=0.5)
    parser.add_argument('--save_model', action='store_true', help='save the model to use later for inference')
    parser.add_argument('--feature_prop', type=str, default='gcn',
                        help='how to propagate ELPH node features. Values are gcn, residual (resGCN) or cat (jumping knowledge networks)')
    # SEAL settings
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_seal_layers', type=int, default=3)
    parser.add_argument('--sortpool_k', type=float, default=0.6)
    parser.add_argument('--label_pooling', type=str, default='add', help='add or mean')
    parser.add_argument('--seal_pooling', type=str, default='edge', help='how SEAL pools features in the subgraph')
    # Subgraph settings
    parser.add_argument('--num_hops', type=int, default=1)
    parser.add_argument('--ratio_per_hop', type=float, default=1.0)
    parser.add_argument('--max_nodes_per_hop', type=int, default=None)
    parser.add_argument('--node_label', type=str, default='drnl')
    parser.add_argument('--max_dist', type=int, default=4)
    parser.add_argument('--max_z', type=int, default=1000,
                        help='the size of the label embedding table. ie. the maximum number of labels possible')
    parser.add_argument('--use_feature', type=str2bool, default=True,
                        help="whether to use raw node features as GNN input")
    parser.add_argument('--use_struct_feature', type=str2bool, default=True,
                        help="whether to use structural graph features as GNN input")
    parser.add_argument('--use_edge_weight', action='store_true',
                        help="whether to consider edge weight in GNN")
    # Training settings
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--adv_lr', type=float, default=0.00001)
    parser.add_argument('--reg_lambda', type=float, default=0.001, help="regularization weight $\lambda$ for the adversary")
    parser.add_argument('--no_intervention', action='store_true', help="turns off or on the intervention loss")
    
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for optimization')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_negs', type=int, default=1, help='number of negatives for each positive')
    parser.add_argument('--train_node_embedding', action='store_true',
                        help="also train free-parameter node embeddings together with GNN")
    parser.add_argument('--propagate_embeddings', action='store_true',
                        help='propagate the node embeddings using the GCN diffusion operator')
    parser.add_argument('--loss', default='bce', type=str, help='bce or auc')
    parser.add_argument('--add_normed_features', dest='add_normed_features', type=str2bool,
                        help='Adds a set of features that are normalsied by sqrt(d_i*d_j) to calculate cosine sim')
    parser.add_argument('--use_RA', type=str2bool, default=False, help='whether to add resource allocation features')
    # SEAL specific args
    parser.add_argument('--dynamic_train', action='store_true',
                        help="dynamically extract enclosing subgraphs on the fly")
    parser.add_argument('--dynamic_val', action='store_true')
    parser.add_argument('--dynamic_test', action='store_true')
    parser.add_argument('--pretrained_node_embedding', type=str, default=None,
                        help="load pretrained node embeddings as additional node features")
    # Testing settings
    parser.add_argument('--reps', type=int, default=1, help='the number of repetition of the experiment to run')
    parser.add_argument('--use_valedges_as_input', action='store_true')
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--eval_metric', type=str, default='hits',
                        choices=('hits', 'mrr', 'auc'))
    parser.add_argument('--K', type=int, default=100, help='the hit rate @K')
    # hash settings
    parser.add_argument('--use_zero_one', type=str2bool,
                        help="whether to use the counts of (0,1) and (1,0) neighbors")
    parser.add_argument('--floor_sf', type=str2bool, default=0,
                        help='the subgraph features represent counts, so should not be negative. If --floor_sf the min is set to 0')
    parser.add_argument('--hll_p', type=int, default=8, help='the hyperloglog p parameter')
    parser.add_argument('--minhash_num_perm', type=int, default=128, help='the number of minhash perms')
    parser.add_argument('--max_hash_hops', type=int, default=2, help='the maximum number of hops to hash')
    # wandb settings
    parser.add_argument('--wandb', action='store_true', help="flag if logging to wandb")
    parser.add_argument('--wandb_offline', dest='use_wandb_offline',
                        action='store_true')  # https://docs.wandb.ai/guides/technical-faq

    parser.add_argument('--wandb_sweep', action='store_true',
                        help="flag if sweeping")  # if not it picks up params in greed_params
    parser.add_argument('--wandb_watch_grad', action='store_true', help='allows gradient tracking in train function')
    parser.add_argument('--wandb_track_grad_flow', action='store_true')

    parser.add_argument('--wandb_entity', default="link-prediction", type=str)
    parser.add_argument('--wandb_project', default="link-prediction", type=str)
    parser.add_argument('--wandb_group', default="testing", type=str, help="testing,tuning,eval")
    parser.add_argument('--wandb_run_name', default=None, type=str)
    parser.add_argument('--wandb_output_dir', default='./wandb_output',
                        help='folder to output results, images and model checkpoints')
    parser.add_argument('--wandb_log_freq', type=int, default=1, help='Frequency to log metrics.')
    parser.add_argument('--wandb_epoch_list', nargs='+', default=[0, 1, 2, 4, 8, 16],
                        help='list of epochs to log gradient flow')
    parser.add_argument('--log_features', action='store_true', help="log feature importance")
    parser.add_argument('--cuda', type=int, default=0)

    global args

    args = parser.parse_args()
    if (args.max_hash_hops == 1) and (not args.use_zero_one):
        print("WARNING: (0,1) feature knock out is not supported for 1 hop. Running with all features")
    print(args)

    # args = initialise_wandb(args)

    if args.wandb_sweep:
        
        sweep_configuration = {
            "name": args.dataset_name,
            "metric": {"name": f'rep{0}_true_positive_rate_disparity_Val', "goal": "minimize"},
            "method": "grid",
            "parameters": {
                'adv_lr':{'values':[0.00001, 0.00005, 0.0001, 0.0005]},
                'reg_lambda':{'values':[0.001, 0.0001, 0.00001]},
            },
        }

        sweep_id = wandb.sweep(sweep_configuration, project="lpfairness", entity="joaopedromattos")
        
        wandb.agent(sweep_id, function=run)
    else:
        run()
