import torch
import torch.nn as nn
from GAE.model import GCN, CN0LinkPredictor, NCNClassifier
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_sparse import SparseTensor
# from GAE.utils import removerepeated, _eval_hits
from torch_geometric.data import Data
from tqdm import tqdm
from loguru import logger
# from GAE.losses import fair_loss
import os



@logger.catch
def main():
    
    processed_graphs_path = '/home/jrm28/fairness/pre_processing_methods/processed_graphs/'
    for dataset in filter(lambda x: "_EDITS.pt" in x, os.listdir(processed_graphs_path)):
        
        logger.info(f"Processing {dataset.strip('_EDITS.pt')} dataset with GAE model.")
        
        # Load a dataset
        data, splits = torch.load(f'{processed_graphs_path}/{dataset}')
        data.edge_index = data.edge_index.to(torch.int64)
        
        data = data.to('cuda:0')
        
        # Example usage
        input_dim = data.x.shape[-1]
        hidden_dim = 64
        latent_dim = 64
        num_nodes = data.num_nodes


        # Create adjacency matrix
        adj = to_dense_adj(data.edge_index, max_num_nodes=data.x.shape[0])[0]
        
        params_model = {
            'in_channels' : input_dim,
            'hidden_channels' : hidden_dim,
            'out_channels' : latent_dim,
            'num_layers' : 3,
            'dropout' : 0.5,
        }
        
        params_classifier = {
            'in_channels' : latent_dim,
            'hidden_channels' : 64,
            'out_channels' : 1,
            'num_layers' : 3,
            'dropout' : 0.5,
        }

        # Initialize the model
        model = GCN(**params_model).to('cuda:0')
        # classifier = CN0LinkPredictor(**params_classifier).to('cuda:0') # gae
        classifier = NCNClassifier(**params_classifier).to('cuda:0') # ncn

        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=0.0001)

        # Training loop
        num_epochs = 500
        for epoch in tqdm(range(num_epochs), desc='Training'):
            model.train()
            classifier.train()
            optimizer.zero_grad()
            
            
            h = model(data.x, data.edge_index)
            # import code
            # code.interact(local={**locals(), **globals()})
            
            train_edges = torch.cat([splits['train']['edge'], splits['train']['edge_neg']]).to('cuda:0')
            train_labels = torch.cat([torch.ones(splits['train']['edge'].size(0)), torch.zeros(splits['train']['edge_neg'].size(0))]).to('cuda:0')
            
            adj = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
            classifier_output = classifier(h, adj, train_edges.t())
            
            loss = criterion(classifier_output.squeeze(-1).sigmoid(), train_labels)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # After training, you can use the model for encoding and link prediction
        model.eval()
        with torch.no_grad():
            z = model(data.x, data.edge_index)
            
            test_edges = torch.cat([splits['test']['edge'], splits['test']['edge_neg']]).to('cuda:0')
            test_outputs = classifier(z, adj, test_edges.t())
            
            # Saving outputs
            torch.save(test_outputs, f'/home/jrm28/fairness/pre_processing_methods/processed_graphs/{dataset.strip("_EDITS.pt")}_EDITSTRAINED_NCN.pt')

            # hits = _eval_hits(pos_preds, neg_preds, 100, 'torch')
            
            # import code
            # code.interact(local={**locals(), **globals()})
            
        

if __name__ == '__main__':
    main()