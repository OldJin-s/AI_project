import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os


class GraphDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = os.path.join(self.data_dir, self.files[index])
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract node features
        nodes = data['nodes']
        node_features = torch.tensor([[node['x'], node['y'], node['z']] for node in nodes], dtype=torch.float)

        # Create adjacency matrix
        num_nodes = len(nodes)
        adj_matrix = torch.zeros((num_nodes, num_nodes))
        for edge in data['edges']:
            start_node = edge['start_node']
            end_node = edge['end_node']
            adj_matrix[start_node, end_node] = 1.0

        return node_features, adj_matrix
