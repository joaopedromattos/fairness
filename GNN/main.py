import torch_geometric
import torch
import argparse
import wandb
from fairness import FairLearner
from model import GNNModel

def parse_args():
    parser = argparse.ArgumentParser(description='GNN')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_channels', type=int, default=2, help='Number of hidden channels')
    parser.add_argument('--hidden_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--conv', type=str, default='GCNConv', help='Type of convolutional layer')
    parser.add_argument('--dataset', type=str, default='facebook', help='Dataset to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--wandb', type=bool, default=True, help='Use wandb')
    parser.add_argument('--project', type=str, default='GNN', help='Project name')
    parser.add_argument('--entity', type=str, default='wandb', help='Entity name')
    parser.add_argument('--name', type=str, default='GNN', help='Run name')
    parser.add_argument('--notes', type=str, default='GNN', help='Run notes')
    parser.add_argument('--tags', type=list, default=['GNN'], help='Run tags')
    return parser.parse_args()


def train():


def test():


def main():
    args = parse_args()
    
    data, splits = load_data(args.dataset)
    
    model = GNNModel(data.num_node_features, args.hidden_channels, args.hidden_layers, getattr(torch_geometric.nn, args.conv))
    fair = FairLearner(data.num_node_features, args.hidden_channels, args.hidden_layers)
    
    
    for i in args.epochs:
        train()
        test()
    
        # save model
        # save results
    