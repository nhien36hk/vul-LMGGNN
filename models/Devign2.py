import torch
import torch.nn as nn
from torch_geometric.nn.conv import GatedGraphConv
from torch_geometric.nn import global_mean_pool
import os

from .layers import Conv
from .AutoEncoder import VectorAutoencoder


class Devign2(nn.Module):
    def __init__(self, gated_graph_conv_args, conv_args, emb_size, device, autoencoder_path, compressed_dim=101):
        super(Devign2, self).__init__()

        # Graph neural network layer
        self.ggnn = GatedGraphConv(**gated_graph_conv_args).to(device)
        
        # AutoEncoder setup
        self.autoencoder = VectorAutoencoder(input_dim=emb_size, compressed_dim=compressed_dim).to(device)
        state = torch.load(autoencoder_path, map_location=device)
        self.autoencoder.load_state_dict(state)
        # Freeze AE parameters
        for p in self.autoencoder.parameters():
            p.requires_grad = False
        self.autoencoder.eval()
        
        # Classifier network
        input_size = gated_graph_conv_args["out_channels"] + compressed_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        ).to(device)

    def forward(self, data):
        x_in, edge_index = data.x, data.edge_index
        
        # Compress input vectors
        with torch.no_grad():
            compressed = self.autoencoder.compress(x_in)
        
        # Graph convolution on compressed vectors
        x_gnn = self.ggnn(compressed, edge_index)
        
        # Combine GNN output with compressed embeddings
        final_node_representation = torch.cat([x_gnn, compressed], dim=1)
        
        # Graph-level pooling
        graph_representation = global_mean_pool(final_node_representation, data.batch)
        
        # Classification with sigmoid activation
        logits = self.classifier(graph_representation)
        probs = torch.sigmoid(logits)
        return probs

    def save(self, path):
        print(path)
        torch.save(self.state_dict(), path)
        print("save!!!!!!")

    def load(self, path):
        self.load_state_dict(torch.load(path))