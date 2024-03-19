import torch
import torch.nn as nn
import torch.nn.functional as F

class AdversaryLearner(nn.Module):
    def __init__(self, input_channels:int, hidden_channels:int=2, hidden_layers:int=2) -> None:
        super(AdversaryLearner, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_layers = hidden_layers
        
        self.lins = nn.ModuleList([nn.Linear(self.input_channels, self.hidden_channels)])
        
        for i in range(hidden_layers):
            self.lins.append(nn.Linear(self.hidden_channels, self.hidden_channels))
            
        # Our adversary classifies considering three possible scenarios:
        # 0 = no sensitive nodes
        # 1 = at least one sensitive node
        # 2 = both sensitive nodes
        self.lins.append(nn.Linear(self.hidden_channels, 1))
        
    def forward(self, batch):
        x = batch
        for i in range(len(self.lins) - 1):
            x = self.lins[i](x)
            x = F.relu(x)
            
        logits = self.lins[-1](x)
        
        return logits
        