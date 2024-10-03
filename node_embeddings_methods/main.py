from FairWalk import FairWalk
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
    parser.add_argument('--model', type=str, default='fair_walk')
    args = parser.parse_args()


    logger.info(f"Processing {args.dataset} dataset with {args.model} model.")
    
    model_class = {'fair_walk': FairWalk}[args.model]

    edge_splits = f"/home/jrm28/fairness/in_processing_methods/NeuralCommonNeighbor/dataset/splits/{args.dataset}.pt"
    data, splits = torch.load(edge_splits)
    
    dataset = {'facebook': Facebook, 'gplus': Google, 'german': German, 'nba': Nba, 'pokec_n': Pokec_n, 'pokec_z': Pokec_z, 'credit': Credit}[args.dataset]()
    adj, features, idx_train, idx_val, idx_test, labels, sens = dataset.adj(), dataset.features(), dataset.idx_train(), dataset.idx_val(), dataset.idx_test(), dataset.labels(), dataset.sens()
    adj = to_torch_sparse_tensor(splits['train']['edge'].t()) # Considering only edges in the training set.      

    logger.info("Initializing model...")
    # Initiate the model (with default parameters).

    if args.model == "fair_walk":
        model = model_class()

        model.fit(adj=adj,
                labels=labels,
                sens=sens,
                idx_train=idx_train.long(),
                dimensions=128,)

        embeddings = model.embs

        
    logger.info("Saving embeddings...")
    torch.save(embeddings, f'/home/jrm28/fairness/node_embeddings_methods/processed_graphs/{args.dataset}_{args.model.upper()}.pt')


if __name__ == '__main__':
    main()