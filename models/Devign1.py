import torch
import torch.nn as nn
from torch_geometric.nn.conv import GatedGraphConv
from torch_geometric.nn import global_mean_pool


class Devign1(nn.Module):
    def __init__(self, gated_graph_conv_args, device):
        super(Devign1, self).__init__()
        
        # Graph neural network layer
        self.ggnn = GatedGraphConv(**gated_graph_conv_args).to(device)
        
        # Classifier network
        self.classifier = nn.Sequential(
            nn.Linear(gated_graph_conv_args["out_channels"], 128), 
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(128, 1) 
        ).to(device)
        
    def forward(self, data):
        # Extract input data
        x_in, edge_index = data.x, data.edge_index
        
        # Graph convolution
        x_gnn = self.ggnn(x_in, edge_index)

        # Graph-level pooling
        graph_representation = global_mean_pool(x_gnn, data.batch)
        
        # Classification
        logits = self.classifier(graph_representation)
        probs = torch.sigmoid(logits)
        
        return probs

    def save(self, path):
        print(path)
        torch.save(self.state_dict(), path)
        print("save!!!!!!")

    def load(self, path):
        self.load_state_dict(torch.load(path))
