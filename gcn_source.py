import torch 
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GCNConv
import torch.optim as optimizer

device = torch.device('cuda:0')

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GraphRNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GraphRNNClassifier, self).__init__()
        # Graph Neural Network
        self.gnn = GCNConv(input_dim, hidden_dim)
        # RNN for temporal dependency
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_dim, 3)  # num_classes=3


    def forward(self, graph_seq, edge_index_seq):
        # graph_seq: [batch_size, seq_len, num_nodes, input_dim]
        batch_size, seq_len, num_nodes, _ = graph_seq.size()
        
        # Process each graph frame in the sequence
        outputs = []
        for t in range(seq_len):
            x_t = graph_seq[:, t]  # [batch_size, num_nodes, input_dim]
            edge_index_t = edge_index_seq[t]  # Assumes static edge index
            
            # Flatten batch and nodes for GNN processing
            x_t = x_t.view(-1, x_t.size(-1))  # [batch_size*num_nodes, input_dim]
            x_t = self.gnn(x_t, edge_index_t)  # [batch_size*num_nodes, hidden_dim]
            
            # Reshape back to [batch_size, num_nodes, hidden_dim]
            x_t = x_t.view(batch_size, num_nodes, -1)
            # Pool over nodes to get graph-level representation
            x_t = x_t.mean(dim=1)  # [batch_size, hidden_dim]
            outputs.append(x_t)

        # Stack graph-level embeddings into sequence
        outputs = torch.stack(outputs, dim=1)  # [batch_size, seq_len, hidden_dim]
        
        # Temporal dependency via RNN
        rnn_out, _ = self.rnn(outputs)  # [batch_size, seq_len, hidden_dim]
        rnn_out = rnn_out[:, -1, :]  # Take the last time step
        
        # Classification
        out = self.fc(rnn_out)  # [batch_size, num_classes]
        return out

