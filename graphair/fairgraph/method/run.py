from Graphair import graphair,aug_module,GCN,GCN_Body,Classifier

import argparse
import torch
import torch_geometric

from types import SimpleNamespace

import scipy.sparse as sp

import time
import os
import datetime

from torch_geometric.data import Data
from torch_geometric.transforms.to_undirected import ToUndirected

class run():
    r"""
    This class instantiates Graphair model and implements method to train and evaluate.
    """

    def __init__(self):
        pass

    def run(self,device,dataset,model='Graphair',epochs=10_000,test_epochs=1_000,
            lr=1e-4,weight_decay=1e-5):
        r""" This method runs training and evaluation for a fairgraph model on the given dataset.
        Check :obj:`examples.fairgraph.Graphair.run_graphair_nba.py` for examples on how to run the Graphair model.

        
        :param device: Device for computation.
        :type device: :obj:`torch.device`

        :param model: Defaults to `Graphair`. (Note that at this moment, only `Graphair` is supported)
        :type model: str, optional
        
        :param dataset: The dataset to train on. Should be one of :obj:`dig.fairgraph.dataset.fairgraph_dataset.POKEC` or :obj:`dig.fairgraph.dataset.fairgraph_dataset.NBA`.
        :type dataset: :obj:`object`
        
        :param epochs: Number of epochs to train on. Defaults to 10_000.
        :type epochs: int, optional

        :param test_epochs: Number of epochs to train the classifier while running evaluation. Defaults to 1_000.
        :type test_epochs: int,optional

        :param lr: Learning rate. Defaults to 1e-4.
        :type lr: float,optional

        :param weight_decay: Weight decay factor for regularization. Defaults to 1e-5.
        :type weight_decay: float, optional

        :raise:
            :obj:`Exception` when model is not Graphair. At this moment, only Graphair is supported.
        """

        # Train script

        dataset_name = dataset.name

        features = dataset.features
        sens = dataset.sens
        adj = dataset.adj
        idx_sens = dataset.idx_sens_train

        # generate model
        if model=='Graphair':
            aug_model = aug_module(features, n_hidden=64, temperature=1).to(device)
            f_encoder = GCN_Body(in_feats = features.shape[1], n_hidden = 64, out_feats = 64, dropout = 0.1, nlayer = 2).to(device)
            sens_model = GCN(in_feats = features.shape[1], n_hidden = 64, out_feats = 64, nclass = 1).to(device)
            classifier_model = Classifier(input_dim=64,hidden_dim=128)
            model = graphair(aug_model=aug_model,f_encoder=f_encoder,sens_model=sens_model,classifier_model=classifier_model, lr=lr,weight_decay=weight_decay,dataset=dataset_name).to(device)
        else:
            raise Exception('At this moment, only Graphair is supported!')
        
        if dataset_name == "NBA":
            # call fit_whole
            st_time = time.time()
            model.fit_whole(epochs=epochs,adj=adj, x=features,sens=sens,idx_sens = idx_sens,warmup=50, adv_epoches=1)
            print("Training time: ", time.time() - st_time)
        else:
            # call fit_batch
            st_time = time.time()
            model.fit_batch(epochs=epochs,adj=adj, x=features,sens=sens,idx_sens = idx_sens,warmup=50, adv_epoches=1)
            print("Training time: ", time.time() - st_time)

        # Test script
        # This will be useful to test whether the graph is fair or not on BUDDY.
        # model.test(adj=adj,features=features,labels=dataset.labels,epochs=test_epochs,idx_train=dataset.idx_train,idx_val=dataset.idx_val,idx_test=dataset.idx_test,sens=sens)
        adj_aug, x_aug, _ = model.get_fair_graph(adj=adj, x=features)
        
        return adj_aug, x_aug



if __name__ == "__main__":
    
    argparse = argparse.ArgumentParser()
    
    argparse.add_argument('--dataset', type=str, default='facebook')
    argparse.add_argument('--device', type=int, default=0)
    argparse.add_argument('--model', type=str, default='Graphair')
    argparse.add_argument('--epochs', type=int, default=150)
    argparse.add_argument('--test_epochs', type=int, default=1_000)
    argparse.add_argument('--lr', type=float, default=1e-4)
    argparse.add_argument('--weight_decay', type=float, default=1e-5)
    args = argparse.parse_args()
    
    dataset_file = {
        'facebook': '/home/jrm28/fairness/NeuralCommonNeighbor/dataset/splits/facebook.pt',
        'gplus': '/home/jrm28/fairness/NeuralCommonNeighbor/dataset/splits/gplus.pt',
    }[args.dataset]
    
    device = f'cuda:{args.device}'
    
    data, split_edge = torch.load(dataset_file)    
    
    adj_matrix = sp.coo_matrix((torch.ones_like(split_edge['train']['edge'].t()[0]).numpy(), (split_edge['train']['edge'].t()[0], split_edge['train']['edge'].t()[1])), shape=(data.num_nodes, data.num_nodes))
    
        
    dataset = SimpleNamespace(name=args.dataset,
                                features=data.x,
                                labels=data.y, # These labels are not used in the Graphair model.
                                sens=data.y,
                                adj=adj_matrix,
                                idx_sens_train=data.y.nonzero().t(),
                                idx_train=torch.arange(data.num_nodes),
                                idx_val=torch.arange(data.num_nodes),
                                idx_test=torch.arange(data.num_nodes))
    
    run = run()
    adj_aug, x_aug = run.run(device, dataset, model='Graphair', epochs=args.epochs, test_epochs=args.test_epochs, lr=args.lr, weight_decay=args.weight_decay)
    adj_aug, x_aug = adj_aug.detach(), x_aug.detach()

    # Masking positive edges in valid and test splits 
    adj_aug[split_edge['valid']['edge'].t()[0], split_edge['valid']['edge'].t()[1]] = 0
    adj_aug[split_edge['test']['edge'].t()[0], split_edge['test']['edge'].t()[1]] = 0
    
    
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists(f"./checkpoint/out/AUGMENTED_{args.dataset}_{args.epochs}_epochs_{timestamp}"):
        os.makedirs(f"./checkpoint/out/AUGMENTED_{args.dataset}_{args.epochs}_epochs_{timestamp}")
    
    data_aug = Data(x=x_aug, y=data.y, edge_index=torch.tensor(adj_aug.nonzero().t()))
    
    data_aug = ToUndirected()(data_aug)
    
    torch.save(data_aug, f"./checkpoint/out/AUGMENTED_{args.dataset}_{args.epochs}_epochs_{timestamp}/splits.pt")
    
    