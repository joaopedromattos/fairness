from NIFTY import NIFTY
from EDITS import EDITS
from datasets import Facebook
from torch_geometric.data import Data
import argparse
import torch
import numpy as np
from torch_geometric.utils import convert

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='facebook')
parser.add_argument('--model', type=str, default='NIFTY')
args = parser.parse_args()


model_class = {'NIFTY': NIFTY, 'EDITS': EDITS}[args.model]

dataset_path = f"/home/jrm28/fairness/NeuralCommonNeighbor/dataset/splits/{args.dataset}.pt"
# splits_path = f"/home/jrm28/fairness/data/edge_splits/{args.dataset}_ncnc.pt"

print("Reading dataset...")
print("Data path:", dataset_path)
data, splits = torch.load(dataset_path)

adj = splits['train']['edge'].t()

print("Converting dataset...")

# Converting the edge list into a sparse torch tensor.
adj = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1]), (data.num_nodes, data.num_nodes))
features = data.x
sens = data.y
sens_idx=-1
node_num = data.num_nodes
idx_train = np.random.choice(
            list(range(node_num)), int(0.8 * node_num), replace=False
        )
idx_val = list(set(list(range(node_num))) - set(idx_train))
idx_test = np.random.choice(idx_val, len(idx_val) // 2, replace=False)
idx_val = list(set(idx_val) - set(idx_test))

idx_train, idx_val, idx_test = torch.tensor(idx_train, dtype=torch.long), torch.tensor(idx_val, dtype=torch.long), torch.tensor(idx_test, dtype=torch.long)


## Getting labels:
if args.dataset == 'facebook':
    temp_features = np.loadtxt("/home/jrm28/fairness/NeuralCommonNeighbor/dataset/ego-facebook/raw/facebook/1684.feat")
    sorted_idx = temp_features[:, 0].argsort()
    temp_features = temp_features[sorted_idx, :]
    labels = torch.tensor(temp_features[:, 101])
    
    # Remove label colum from features
    features = torch.cat((features[:, :101], features[:, 104:]), dim=1)
    # import code
    # code.interact(local={**locals(), **globals()})
    

print("Initializing model...")
# Initiate the model (with default parameters).

if args.model == 'NIFTY':
    
    features = torch.cat((features, sens.view(-1, 1)), dim=1)
    
    model = model_class(adj, features, labels, idx_train, idx_val, idx_test, sens, sens_idx)
    
    print("Training start...")
    # Train the model.
    model.fit()
    
    print("Inference start...")
    # Evaluate the model.
    emb, results = model.predict()
    
    data.x = emb
    
elif args.model == "EDITS":
    model = model_class(features)
    
    model.fit(adj,
        features,
        sens,
        idx_train,
        idx_val)
   
    edge_index, x = convert.from_scipy_sparse_matrix(model.adj1)[0], model.X_debiased
    
    data.edge_index, data.x = edge_index, x

torch.save((data, splits), f'/home/jrm28/fairness/NeuralCommonNeighbor/dataset/splits/{args.dataset}_{args.model.lower()}.pt')
