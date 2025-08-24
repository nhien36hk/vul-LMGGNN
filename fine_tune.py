import os
import numpy as np
import torch
import torch as th
import torch.nn.functional as F
import torch.utils.data as Data
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from models.GraphCodeBERT import GraphCodeBertClassifier
from utils.figure.plot import plot_confusion_matrix, plot_roc_curve
from utils.data.datamanager import read, train_val_test_split

torch.manual_seed(2020)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = th.device('cpu')
gpu = th.device('cuda:0')

def encode_input(text, tokenizer, max_length: int = 512):
    input = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
    return input.input_ids, input.attention_mask

class CustomDataset(Data.Dataset):
    def __init__(self, input_ids, attention_mask, label, nums):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label = label
        self.num = nums

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index], self.label[index], self.num[index]

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epoch}")
    for batch_idx, batch in progress_bar:
        x1, x2, y, num = batch
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        y_pred = model(x1, x2)
        model.zero_grad()
        loss = F.cross_entropy(y_pred, y.squeeze())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{running_loss:.6f}"})

def val(model, device, test_loader, figure_save_path: str = ''):
    model.eval()
    test_loss = 0.0
    y_true = []
    y_pred = []

    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Validating")
    for batch_idx, batch in progress_bar:
        x1, x2, y, num = batch
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        with torch.no_grad():
            y_ = model(x1, x2)
        test_loss += F.cross_entropy(y_, y.squeeze()).item()
        pred = y_.max(-1, keepdim=True)[1]
        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
        progress_bar.set_postfix({"loss": f"{test_loss:.4f}"})

    test_loss /= len(test_loader)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(figure_save_path, 'confusion_matrix_finetune_val.png'),
        title='Confusion Matrix - Fine-tune Model (Validation)'
    )

    print('Valid set: Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(
        test_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100))

    return accuracy, precision, recall, f1

def test(model, device, test_loader, figure_save_path: str = ''):
    model.eval()
    test_loss = 0.0
    y_true = []
    y_pred = []
    y_probs = []
    num_list = []

    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing")
    for batch_idx, batch in progress_bar:
        x1, x2, y, num = batch
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        with torch.no_grad():
            y_ = model(x1, x2)
        test_loss += F.cross_entropy(y_, y.squeeze()).item()
        pred = y_.max(-1, keepdim=True)[1]
        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
        y_probs.extend(torch.softmax(y_, dim=1).cpu().numpy()[:, 1])
        num_list.extend(num.cpu().numpy())
        progress_bar.set_postfix({"loss": f"{test_loss:.4f}"})

    test_loss /= len(test_loader)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(figure_save_path, 'confusion_matrix_finetune_test.png'),
        title='Confusion Matrix - Fine-tune Model (Test)'
    )
    plot_roc_curve(
        y_true, y_probs,
        save_path=os.path.join(figure_save_path, 'roc_curve_finetune_test.png'),
        title='ROC Curve - Fine-tune Model (Test)'
    )
    results_array = np.column_stack((y_true, y_pred, y_probs, num_list))
    header_text = "True label, Predicted label, Predicted Probability"
    np.savetxt(os.path.join(figure_save_path, 'results_finetune_test.txt'), results_array, fmt='%1.6f', delimiter='\t', header=header_text)

    print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(
        test_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100))

    return accuracy, precision, recall, f1
    

if  __name__ == '__main__':
    raw_data = read('data/raw/', 'dataset.json')
    print(f"Total samples after filtering: {len(raw_data)}")
    train_dataset, test_dataset, val_dataset = train_val_test_split(raw_data, shuffle=True)

    train_data = train_dataset.dataset["func"].tolist()
    val_data = val_dataset.dataset["func"].tolist()
    test_data = test_dataset.dataset["func"].tolist()

    train_labels = torch.tensor(train_dataset.dataset["target"].values).to(torch.int64)
    val_labels = torch.tensor(val_dataset.dataset["target"].values).to(torch.int64)
    test_labels = torch.tensor(test_dataset.dataset["target"].values).to(torch.int64)

    train_nums = train_dataset.dataset.index.tolist()
    val_nums = val_dataset.dataset.index.tolist()
    test_nums = test_dataset.dataset.index.tolist()

    batch_size = 128
    tokenizer = AutoTokenizer.from_pretrained('microsoft/graphcodebert-base')

    print("Encoding input data...")
    train_input_ids, train_attention_mask = encode_input(train_data, tokenizer)
    val_input_ids, val_attention_mask = encode_input(val_data, tokenizer)
    test_input_ids, test_attention_mask = encode_input(test_data, tokenizer)

    train_input_ids = train_input_ids.to(device)
    val_input_ids = val_input_ids.to(device)
    test_input_ids = test_input_ids.to(device)
    train_attention_mask = train_attention_mask.to(device)
    val_attention_mask = val_attention_mask.to(device)
    test_attention_mask = test_attention_mask.to(device)
    train_labels = train_labels.to(device)
    val_labels = val_labels.to(device)
    test_labels = test_labels.to(device)

    train_ds = CustomDataset(train_input_ids, train_attention_mask, train_labels, train_nums)
    val_ds = CustomDataset(val_input_ids, val_attention_mask, val_labels, val_nums)
    test_ds = CustomDataset(test_input_ids, test_attention_mask, test_labels, test_nums)

    train_loader = Data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = Data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = Data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = GraphCodeBertClassifier().to(device)
    if hasattr(model.backbone, 'gradient_checkpointing_enable'):
        model.backbone.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    NUM_EPOCHS = 10
    SAVE_PREFIX = 'GraphCodeBERT_finetune_'

    print(f"Starting training for {NUM_EPOCHS} epochs with batch_size={batch_size}...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch)
        acc, precision, recall, f1 = val(model, device, val_loader)
        model_name = SAVE_PREFIX + f'epoch_{epoch}.pth'
        torch.save(model.state_dict(), model_name)
        print("acc is: {:.4f}\n".format(acc))

    print("Training completed. Testing best model...")
    model_test = GraphCodeBertClassifier().to(device)
    model_test.load_state_dict(torch.load(SAVE_PREFIX + "epoch_7.pth"))
    test(model_test, device, test_loader)
