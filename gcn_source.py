import torch 
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import Sequential, GCNConv
import torch.optim as optimizer

device = torch.device('cuda:0')

class PoseGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PoseGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x