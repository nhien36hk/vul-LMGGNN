import torch
import torch.nn as nn
from transformers import AutoModel

class GraphCodeBertClassifier(nn.Module):
    def __init__(self, model_name: str = 'microsoft/graphcodebert-base', num_classes: int = 2):
        super(GraphCodeBertClassifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = AutoModel.from_pretrained(model_name).to(self.device)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size, num_classes).to(self.device)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device))
        last_hidden = outputs.last_hidden_state  # [B, L, H]
        cls_repr = last_hidden[:, 0]  # Use first token as CLS for RoBERTa/GraphCodeBERT
        x = self.dropout(cls_repr)
        logits = self.fc(x)
        return logits

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path)) 