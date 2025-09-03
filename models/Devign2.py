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

        self.ggnn = GatedGraphConv(**gated_graph_conv_args).to(device)
        
        # Thay thế Conv phức tạp bằng MLP đơn giản
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
        
        self.device = device

        self.autoencoder = VectorAutoencoder(input_dim=emb_size, compressed_dim=compressed_dim).to(device)
        state = torch.load(autoencoder_path, map_location=device)
        self.autoencoder.load_state_dict(state)
        # Freeze AE parameters
        for p in self.autoencoder.parameters():
            p.requires_grad = False
        self.autoencoder.eval()

    def forward(self, data):
        x_in, edge_index = data.x, data.edge_index
        
        # Nén vector input
        with torch.no_grad():
            compressed = self.autoencoder.compress(x_in)
        
        # Chạy GGNN trên vector nén
        x_gnn = self.ggnn(compressed, edge_index)
        
        # Nối output của GNN và embedding nén lại (Đã đúng)
        final_node_representation = torch.cat([x_gnn, compressed], dim=1)
        
        graph_representation = global_mean_pool(final_node_representation, data.batch)
        
        logits = self.classifier(graph_representation)
        
        # Sigmoid để có output [0, 1] (Đã đúng)
        probs = torch.sigmoid(logits)
        return probs

    def save(self, path):
        print(path)
        torch.save(self.state_dict(), path)
        print("save!!!!!!")

    def load(self, path):
        self.load_state_dict(torch.load(path))

