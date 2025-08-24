import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn.conv import GatedGraphConv

from transformers import AutoModel, AutoTokenizer
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel

from .layers import Conv, encode_input


class BertGGCN(nn.Module):
    def __init__(self, gated_graph_conv_args, conv_args, emb_size, device, k=0.6):
        super(BertGGCN, self).__init__()
        
        self.in_proj = nn.Linear(emb_size, gated_graph_conv_args["out_channels"]).to(device)
        self.k = k
        self.ggnn = GatedGraphConv(**gated_graph_conv_args).to(device)
        self.conv = Conv(**conv_args,
                         fc_1_size=gated_graph_conv_args["out_channels"] + emb_size,
                         fc_2_size=gated_graph_conv_args["out_channels"]).to(device)
        self.nb_class = 2
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.bert_model = RobertaModel.from_pretrained("microsoft/codebert-base").to(device)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = torch.nn.Linear(self.feat_dim, self.nb_class).to(device)
        self.device = device

    def forward(self, data):
        x_in, edge_index, func_text = data.x, data.edge_index, data.func
        x_gnn_input = self.in_proj(x_in)
        x_gnn = self.ggnn(x_gnn_input, edge_index)
        x_conv = self.conv(x_gnn, x_in)
        input_ids, attention_mask = encode_input(func_text, self.tokenizer)
        cls_feats = self.bert_model(input_ids.to(self.device), attention_mask.to(self.device))[0][:, 0]
        cls_logit = self.classifier(cls_feats.to(self.device))  # [B, 2]
        # Convert 2-class to 1-class
        s_cls = cls_logit[:, 1] - cls_logit[:, 0]  # [B]
        
        # Normalize cả 2 về [0, 1] bằng sigmoid
        x_conv_norm = torch.sigmoid(x_conv)  # [0, 1]
        s_cls_norm = torch.sigmoid(s_cls)    # [0, 1]
        
        # Combine với fixed weights
        pred = x_conv_norm * self.k + s_cls_norm.unsqueeze(-1) * (1 - self.k)
        return pred

    def save(self, path):
        print(path)
        torch.save(self.state_dict(), path)
        print("save!!!!!!")

    def load(self, path):
        self.load_state_dict(torch.load(path))

