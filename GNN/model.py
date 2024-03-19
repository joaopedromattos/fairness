import torch_geometric
import torch

class GNNModel(torch.nn.Module):
    def __init__(self, input_channels:int, hidden_channels:int=2, hidden_layers:int=2, conv=torch_geometric.nn.GCNConv) -> None:
        super(GNNModel, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_layers = hidden_layers
        self.conv_type = conv
        
        self.convs = torch.nn.ModuleList([self.conv_type(self.input_channels, self.hidden_channels)])
        
        for i in range(hidden_layers):
            self.convs.append(self.conv_type(self.hidden_channels, self.hidden_channels))

        self.lin = torch.nn.Linear(self.hidden_channels, 1)
        
    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            
        x = self.lin(x)
        
        return x