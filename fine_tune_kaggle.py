import os

import torch
import torch.utils.data as Data
from transformers import AutoTokenizer

from models.GraphCodeBERT import GraphCodeBertClassifier
from utils.data.datamanager import read, train_val_test_split
from fine_tune import train, val, test, encode_input, CustomDataset

def run_kaggle_finetune(
    raw_dir: str = '/kaggle/input/lm-train/LMTrain/data/raw/',
    json_file: str = 'dataset.json',
    output_model_dir: str = '/kaggle/working/trained_models/',
    model_filename: str = 'graphcodebert_finetune.pt',
    figure_save_path: str = '/kaggle/working/figures/',
    batch_size: int = 128,
    epochs: int = 10,
    lr: float = 2e-5,
    weight_decay: float = 1e-4,
    max_length: int = 128,
    model_name: str = 'microsoft/graphcodebert-base'
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_model_dir, exist_ok=True)
    os.makedirs(figure_save_path, exist_ok=True)

    raw_data = read(raw_dir, json_file)
    print(f"Loaded raw dataset: {len(raw_data)} rows from {os.path.join(raw_dir, json_file)}")

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

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Encoding input data...")
    train_input_ids, train_attention_mask = encode_input(train_data, tokenizer, max_length)
    val_input_ids, val_attention_mask = encode_input(val_data, tokenizer, max_length)
    test_input_ids, test_attention_mask = encode_input(test_data, tokenizer, max_length)

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

    if torch.cuda.device_count() > 1:
        model = GraphCodeBertClassifier(model_name=model_name)
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = GraphCodeBertClassifier(model_name=model_name).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_f1 = 0.0
    best_path = os.path.join(output_model_dir, model_filename)

    print(f"Start training: epochs={epochs}, batch_size={batch_size}, max_length={max_length}")
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        acc, precision, recall, f1 = val(model, device, val_loader, figure_save_path)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_path)
        print(f"Epoch {epoch}: acc={acc:.4f} f1={f1:.4f} best_f1={best_f1:.4f}")

    print("Training finished. Testing best model...")
    best_model = GraphCodeBertClassifier(model_name=model_name).to(device)
    best_model.load_state_dict(torch.load(best_path, map_location=device))
    test(best_model, device, test_loader, figure_save_path)