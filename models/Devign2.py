import torch
import torch.nn as nn
from torch_geometric.nn.conv import GatedGraphConv
import os

from .layers import Conv
from .AutoEncoder import VectorAutoencoder


class Devign2(nn.Module):
    def __init__(self, gated_graph_conv_args, conv_args, emb_size, device, autoencoder_path):
        super(Devign2, self).__init__()

        self.ggnn = GatedGraphConv(**gated_graph_conv_args).to(device)
        self.conv = Conv(**conv_args,
                         fc_1_size=gated_graph_conv_args["out_channels"] + emb_size,
                         fc_2_size=gated_graph_conv_args["out_channels"]).to(device)
        self.device = device

        self.autoencoder = VectorAutoencoder(input_dim=emb_size, compressed_dim=101).to(device)
        state = torch.load(autoencoder_path, map_location=device)
        self.autoencoder.load_state_dict(state)
        # Freeze AE parameters
        for p in self.autoencoder.parameters():
            p.requires_grad = False
        self.autoencoder.eval()

    def forward(self, data):
        x_in, edge_index = data.x, data.edge_index
        with torch.no_grad():
            compressed = self.autoencoder.compress(x_in)
        x_gnn_input = compressed
        x_gnn = self.ggnn(x_gnn_input, edge_index)
        x_conv = self.conv(x_gnn, x_in)
        x_conv_norm = torch.sigmoid(x_conv)  # [0, 1]

        return x_conv_norm

    def save(self, path):
        print(path)
        torch.save(self.state_dict(), path)
        print("save!!!!!!")

    def load(self, path):
        self.load_state_dict(torch.load(path))

