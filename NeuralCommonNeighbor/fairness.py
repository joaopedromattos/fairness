import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch.nn import BatchNorm1d 


class FairLearner(nn.Module):
    def __init__(self, input_channels:int, hidden_channels:int=2, hidden_layers:int=2, out_channels:int=1) -> None:
        super(FairLearner, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_layers = hidden_layers
        self.out_channels = out_channels
        
        self.lins = nn.ModuleList([nn.Linear(self.input_channels, self.hidden_channels)])
        
        self.layer_norms = nn.ModuleList([nn.BatchNorm1d(self.hidden_channels)])
        
        for i in range(hidden_layers):
            self.lins.append(nn.Linear(self.hidden_channels, self.hidden_channels))
            self.layer_norms.append(nn.BatchNorm1d(self.hidden_channels))
            
        # Our adversary classifies considering three possible scenarios:
        # 0 = no sensitive nodes
        # 1 = at least one sensitive node
        # 2 = both sensitive nodes
        self.lins.append(nn.Linear(self.hidden_channels, self.out_channels))
        
    def forward(self, batch):
        x = batch
        for lin, layer_norm in zip(self.lins[:-1], self.layer_norms):
            x = lin(x)
            x = layer_norm(x)
            x = F.leaky_relu(x)
            
        x = self.lins[-1](x)
        
        return x



# class FairLearner(nn.Module):
#     def __init__(self, input_channels:int, hidden_channels:int=2, hidden_layers:int=2, out_channels:int=1) -> None:
#         super(FairLearner, self).__init__()
#         self.input_channels = input_channels
#         self.hidden_channels = hidden_channels
#         self.hidden_layers = hidden_layers
#         self.out_channels = out_channels
        
#         self.lins = nn.ModuleList([nn.Linear(self.input_channels, self.hidden_channels)])
        
#         self.layer_norms = nn.ModuleList([nn.BatchNorm1d(self.hidden_channels)])
        
#         for i in range(hidden_layers):
#             self.lins.append(nn.Linear(self.hidden_channels, self.hidden_channels))
#             # self.layer_norms.append(nn.BatchNorm1d(self.hidden_channels))
            
#         # Our adversary classifies considering three possible scenarios:
#         # 0 = no sensitive nodes
#         # 1 = at least one sensitive node
#         # 2 = both sensitive nodes
#         self.lins.append(nn.Linear(self.hidden_channels, self.out_channels))
        
#     def forward(self, batch):
#         x = batch
#         for lin in self.lins[:-1]:
#             x = lin(x)
#             x = F.leaky_relu(x)
            
#         x = self.lins[-1](x)
        
#         return x
        




class FairLearner_GNN(torch.nn.Module):
    
    def __init__(self, input_channels:int, hidden_channels:int=2, hidden_layers:int=2, out_channels:int=1):
        super(FairLearner_GNN, self).__init__()
        # layer = GCNConv
        layer = GATConv
        self.num_layers = hidden_layers
        
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        self.convs.append(layer(input_channels, hidden_channels))
        self.batch_norms.append(BatchNorm1d(hidden_channels))
        
        for _ in range(hidden_layers - 2):
            self.convs.append(layer(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm1d(hidden_channels))
        
        self.convs.append(layer(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.leaky_relu(x)
            # x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x
        