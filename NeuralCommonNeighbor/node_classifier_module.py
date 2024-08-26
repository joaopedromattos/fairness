import torch.nn as nn
from tqdm import tqdm

class NodeClassifierModule(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout) -> None:
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
        
    def fit(self, x, y, optimizer, criterion, epochs, log=None):
        self.model.train()
        for epoch in tqdm(range(epochs)):
            optimizer.zero_grad()
            y_pred = self.model(x)
            loss = criterion(y_pred.softmax(axis=1), y)
            loss.backward()
            optimizer.step()
            
            if log:
                log.log({
                    "epoch_step" : epoch,
                    "loss" : loss.item(),
                })
            
    def predict(self, x):
        return self.model(x)
    
    def evaluation(self, preds, target, metrics, task, num_classes):
        return metrics(preds, target, task=task, num_classes=num_classes)

        
        