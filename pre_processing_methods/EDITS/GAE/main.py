import torch
import torch.nn as nn
from model import GraphAutoencoder
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from utils import removerepeated, _eval_hits
from torch_geometric.data import Data
from tqdm import tqdm
from loguru import logger
from losses import fair_loss




@logger.catch
def main():
    # Load a dataset
    data, splits = torch.load('/home/jrm28/fairness/NeuralCommonNeighbor/dataset/splits/facebook.pt')
    
    # Example usage
    input_dim = data.x.shape[-1]
    hidden_dim = 64
    latent_dim = 64
    num_nodes = data.num_nodes


    # Create adjacency matrix
    adj = to_dense_adj(data.edge_index)[0]

    # Initialize the model
    model = GraphAutoencoder(input_dim, hidden_dim, latent_dim)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    for epoch in tqdm(range(num_epochs), desc='Training'):
        model.train()
        optimizer.zero_grad()
        
        adj_reconstructed = model(data.x, data.edge_index)
        loss = criterion(adj_reconstructed, adj)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # After training, you can use the model for encoding and link prediction
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        adj_reconstructed = model.decode(z)
        
        pos_preds = adj_reconstructed[splits['test']['edge'][:, 0], splits['test']['edge'][:, 1]]
        neg_preds = adj_reconstructed[splits['test']['edge_neg'][:, 0], splits['test']['edge_neg'][:, 1]]

        hits = _eval_hits(pos_preds, neg_preds, 100, 'torch')
        
        import code
        code.interact(local={**locals(), **globals()})
        
        

if __name__ == '__main__':
    main()