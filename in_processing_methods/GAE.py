import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GraphAutoencoder, self).__init__()
        
        # Encoder
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)
        
        # Decoder (inner product decoder)
    
    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)
    
    def decode(self, z):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj)
    
    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        adj_reconstructed = self.decode(z)
        return adj_reconstructed