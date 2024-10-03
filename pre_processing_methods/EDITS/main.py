from EDITS import EDITS
from EDITS_batched import EDITS
from torch_geometric.data import Data
from utils import to_torch_sparse_tensor
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.utils import convert
from torch_sparse import SparseTensor
from loguru import logger
from datasets import Facebook, Google, German, Nba, Pokec_n, Pokec_z, Credit



def mask_graphair(old_split_edge, new_edge_index):
    logger.info("Masking edges...")
    mask = torch.zeros(new_edge_index.size(1), dtype=torch.bool).cpu()
    for edge in tqdm(old_split_edge):
        mask |= ((edge[0] == new_edge_index).any(0) & (edge[1] == new_edge_index).any(0))           
    return mask


@logger.catch
def main():
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='facebook')
    parser.add_argument('--model', type=str, default='EDITS')
    args = parser.parse_args()


    logger.info(f"Processing {args.dataset} dataset with {args.model} model.")
    
    model_class = {'EDITS': EDITS}[args.model]

    edge_splits = f"/home/jrm28/fairness/in_processing_methods/NeuralCommonNeighbor/dataset/splits/{args.dataset}.pt"
    data, splits = torch.load(edge_splits)
    
    dataset = {'facebook': Facebook, 'google': Google, 'german': German, 'nba': Nba, 'pokec_n': Pokec_n, 'pokec_z': Pokec_z, 'credit': Credit}[args.dataset]()
    adj, features, idx_train, idx_val, idx_test, labels, sens = dataset.adj(), dataset.features(), dataset.idx_train(), dataset.idx_val(), dataset.idx_test(), dataset.labels(), dataset.sens()
    adj = to_torch_sparse_tensor(splits['train']['edge'].t()) # Considering only edges in the training set.      

    logger.info("Initializing model...")
    # Initiate the model (with default parameters).

    if args.model == "EDITS":
        model = model_class(features)

        model.fit(adj,
            features,
            sens,
            idx_train,
            idx_val, device='cuda:0')

        import code
        code.interact(local=locals())
        edge_index, x = convert.from_scipy_sparse_matrix(model.adj1)[0], model.X_debiased

        # Remove any validation or test edges that are not in the training set to avoid any backpropagation using these edges.
        mask = mask_graphair(torch.cat([splits['valid']['edge_neg'], splits['test']['edge_neg'], splits['valid']['edge'], splits['test']['edge']]).cpu(), edge_index.cpu())
        edge_index = edge_index[:, mask]

        data.edge_index, data.adj_t, data.x = edge_index.float(), SparseTensor.from_edge_index(edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)).float(), x.float()

        
    logger.info("Saving processed data...")
    torch.save((data, splits), f'/home/jrm28/fairness/pre_processing_methods/processed_graphs/{args.dataset}_{args.model.upper()}.pt')


if __name__ == '__main__':
    main()