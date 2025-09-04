import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BertClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_classes=2):
        super(BertClassifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes).to(self.device)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device))
        pooled_output = outputs.pooler_output

        x = self.dropout(pooled_output.to(self.device))
        logits = self.fc(x)

        return logits

    def save(self, path):
        print(path)
        torch.save(self.state_dict(), path)
        print("save!!!!!!")

    def load(self, path):
        self.load_state_dict(torch.load(path))