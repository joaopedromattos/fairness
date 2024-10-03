import torch
import torch.functional as F
import torch.nn as nn

class NIFTY(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NIFTY, self).__init__()
        
        # Encoder
        self.projection_layer1 = nn.Linear(input_dim, hidden_dim)
        self.projection_layer2 = nn.Linear(hidden_dim, hidden_dim)
    
    def NIFTY_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        
        def D(x1, x2):  # negative cosine similarity
            return -F.cosine_similarity(x1, x2.detach(), dim=-1).mean()

        # projector
        p1 = self.projection_layer1(z1)
        p2 = self.projection_layer1(z2)

        # predictor
        h1 = self.projection_layer2(p1)
        h2 = self.projection_layer2(p2)

        l1 = D(h1, p2) / 2
        l2 = D(h2, p1) / 2

        return (l1 + l2)
    
    
fair_loss = {
    'NIFTY': NIFTY
}