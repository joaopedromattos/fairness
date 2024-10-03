from torch_geometric.data import Data
from utils import get_dataset
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.utils import convert
from torch_sparse import SparseTensor
from loguru import logger
from NIFTY import NIFTY
from FairVGNN import FairVGNN
from FairGNN import FairGNN
from GAE import GraphAutoencoder
from DELTR import DELTR


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
    parser.add_argument('--model', type=str, default='gae')
    parser.add_argument('--fair_model', type=str, default='fair_walk')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--exposure_coeff', type=float, default=50_000)
    
    args = parser.parse_args()


    logger.info(f"Processing {args.dataset} dataset with {args.fair_model} model.")
    
    model = {
        'gae' : {
            'encoder': 'gcn',
            'decoder': 'gae',
        },
        'ncn' : {
            'encoder': 'gcn',
            'decoder': 'ncn',
        },
        'seal':{
            'encoder': 'seal',
            'decoder': 'seal',            
        }
    }[args.model]
    
    fair_model = {'nifty': NIFTY, "deltr": DELTR, "fair_gnn": FairGNN, "fair_vgnn": FairVGNN}[args.fair_model]

    adj, features, idx_train, idx_val, idx_test, labels, sens, data, splits = get_dataset(args.dataset)

    logger.info("Initializing model...")
    # Initiate the model (with default parameters).

    if args.fair_model == "nifty":
        
        params = {
            'adj':adj,
            'features':features,
            'labels':labels,
            'idx_train':idx_train.long(),
            'idx_val':idx_val.long(),
            'idx_test':idx_test.long(),
            'sens':sens,
            'sens_idx': -1,
            'num_hidden':128,
            'num_proj_hidden':128,
            'encoder': model['encoder'],
            'decoder': model['decoder'],
            'edge_splits': splits,
            'dataset_name': args.dataset,
            'device': 'cuda:1',
            # 'device': 'cpu',
        }
        
        fair_model = fair_model(**params)
        
        fair_model.fit(epochs=500)

        output, counter_output, noisy_output = fair_model.predict()
        torch.save((output, counter_output, noisy_output), f'/home/jrm28/fairness/in_processing_methods/outputs/{args.dataset}_{args.fair_model.upper()}_{args.model.upper()}.pt')
        
    elif args.fair_model == 'deltr':
        params = {
            'adj':adj,
            'features':features,
            'labels':labels,
            'idx_train':idx_train.long(),
            'idx_val':idx_val.long(),
            'idx_test':idx_test.long(),
            'sens':sens,
            'sens_idx': -1,
            'num_hidden':128,
            'num_proj_hidden':128,
            'encoder': model['encoder'],
            'decoder': model['decoder'],
            'edge_splits': splits,
            'dataset_name': args.dataset,
            'device': args.device,
            'exposure_coeff': args.exposure_coeff,
            'lr': args.lr
            # 'device': 'cpu',
        }
        
        fair_model = fair_model(**params)
        
        # fair_model = torch.compile(fair_model)
        
        fair_model.fit(epochs=500)

        output = fair_model.predict()
        # import code
        # code.interact(local={**locals(), **globals()})
        torch.save(output, f'/home/jrm28/fairness/in_processing_methods/outputs/{args.dataset}_{args.fair_model.upper()}_{args.model.upper()}.pt')
        
    elif args.fair_model == 'fair_gnn':
        
        features = features[:, :-1] # remove the sensitive attribute
        params = {
            'nfeat':features.shape[1],
            'encoder': model['encoder'],
            'decoder': model['decoder'],
            'edge_splits':splits,
            'sim_coeff':0.6,
            'n_order':10,
            'subgraph_size':30,
            'acc':0.69,
            'epoch':2000,
        }
        
        fair_model = fair_model(**params)
        
        params = {
            'g': adj,
            'features': features, 
            'labels': labels,
            'idx_train':idx_train.long(),
            'idx_val':idx_val.long(),
            'idx_test':idx_test.long(),
            'sens':sens,
            'device':'cuda:1',
            
        }
        
        fair_model.fit(**params)

        output = fair_model.predict()
        
        logger.info(f"[{args.fair_model}] Saving outputs...")
        torch.save(output, f'/home/jrm28/fairness/in_processing_methods/outputs/{args.dataset}_{args.fair_model.upper()}_{args.model.upper()}.pt')


        
    elif args.fair_model == 'fair_vgnn':
        
        
        fair_model = fair_model()
        
        params = {
            'adj': adj,
            'feats': features, 
            'labels': labels,
            'idx_train':idx_train.long(),
            'idx_val':idx_val.long(),
            'idx_test':idx_test.long(),
            'sens':sens,
            'sens_idx': -1,
            'device':'cuda:2',
            'edge_splits': splits,
            'encoder': model['encoder'],
            'classifier_name' : model['decoder']
        }
        
        test_output = fair_model.fit(**params)

        # output = fair_model.predict()
    
        
        logger.info(f"[{args.fair_model}] Saving outputs...")
        torch.save(test_output, f'/home/jrm28/fairness/in_processing_methods/outputs/{args.dataset}_{args.fair_model.upper()}_{args.model.upper()}.pt')


if __name__ == '__main__':
    main()