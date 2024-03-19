import numpy as np
import pandas as pd
import torch_geometric
import networkx as nx
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets.planetoid import Planetoid
from torch_geometric.transforms.to_undirected import ToUndirected
import torch

from torch_geometric.datasets import SNAPDataset

dataset = SNAPDataset(root='/home/jrm28/fairness/subgraph_sketching-original/dataset/gplus_test', name='ego-gplus')