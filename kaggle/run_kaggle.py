import os
import torch

import configs
from models.Devign1 import Devign1
from models.Devign2 import Devign2
from models.Devign2Linear import Devign2Linear
from models.LMGNN import BertGGCN
from run import train, validate, plot_validation_loss
from test import test
from utils.data.helper import loads as load_datasets
from utils.data.input import train_val_test_split, load_split_datasets
from utils.data.helper import check_split_exists


def run_kaggle_train(
    data_dir: str = '/kaggle/input/lm-train/LMTrain/data/',
    kaggle_working_dir: str = '/kaggle/working/',
    model_filename: str = 'devign2.pt',
    autoencoder_filename: str = 'autoencoder.pt',
    finetune_filename: str = 'graphcodebert_finetune.pt',
    hugging_path_filename: str = 'graphcodebert_finetune_hf',
    batch_size: int = 12,
    epochs: int = 30,
    k: float = 0.6,
    mode_lm: bool = True,
    autoencoder_path: str = "",
    split_dir: str = "",
):
    output_model_dir = os.path.join(kaggle_working_dir, 'trained_models')
    figure_save_path = os.path.join(kaggle_working_dir, 'figures')

    input_dir = os.path.join(data_dir, 'input')
    finetune_file = os.path.join(data_dir, 'model', finetune_filename)
    
    hugging_path = os.path.join(output_model_dir, hugging_path_filename)
    best_path = os.path.join(output_model_dir, model_filename)

    os.makedirs(output_model_dir, exist_ok=True)
    os.makedirs(figure_save_path, exist_ok=True)

    # Load merged dataset using project utility
    input_dataset = load_datasets(input_dir)
    print(f"Loaded input dataset of size: {len(input_dataset)} from {input_dir}")

    # Split into InputDataset wrappers using project utility
    proc_config = configs.Process()
    shuffle = proc_config.shuffle
    if check_split_exists(split_dir):
        print(f"Loading split dataset from {split_dir}")
        train_ds, val_ds, test_ds, test_short_ds, test_long_ds = load_split_datasets(split_dir, input_dataset)
    else:
        train_ds, val_ds, test_ds, test_short_ds, test_long_ds = train_val_test_split(input_dataset, shuffle=shuffle, save_path=output_model_dir)

    # Build DataLoaders
    train_loader = train_ds.get_loader(batch_size, shuffle=True)
    val_loader = val_ds.get_loader(batch_size, shuffle=False)
    test_loader = test_ds.get_loader(batch_size, shuffle=False)
    test_short_loader = test_short_ds.get_loader(batch_size, shuffle=False)
    test_long_loader = test_long_ds.get_loader(batch_size, shuffle=False)

    # Model & Optimizer
    bertggnn = configs.BertGGNN()
    gated_graph_conv_args = bertggnn.model["gated_graph_conv_args"]
    conv_args = bertggnn.model["conv_args"]
    emb_size = bertggnn.model["emb_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if mode_lm:
        model = BertGGCN(gated_graph_conv_args, conv_args, emb_size, device, k, hugging_path, finetune_file).to(device)
        best_model = BertGGCN(gated_graph_conv_args, conv_args, emb_size, device, k, hugging_path, finetune_file).to(device)
    else:
        autoencoder_path = "data/model/autoencoder.pt"
        compressed_dim = bertggnn.model["compressed_dim"]
        model = Devign2Linear(gated_graph_conv_args, conv_args, emb_size, device).to(device)
        best_model = Devign2Linear(gated_graph_conv_args, conv_args, emb_size, device).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=bertggnn.learning_rate, 
        weight_decay=bertggnn.weight_decay
    )

    # Train loop
    best_f1 = 0.0
    losses = []
    
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        loss, acc, precision, recall, f1 = validate(model, device, val_loader, epoch)
        losses.append(loss)
        if best_f1 < f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_path)

    plot_validation_loss(losses, os.path.join(figure_save_path, 'validation_loss.png'))
    print(f"Training finished. Best F1: {best_f1:.4f}")

    # Load best and evaluate on test
    best_model.load_state_dict(torch.load(best_path, map_location=device))
    print("Testing full")
    test(best_model, device, test_loader, os.path.join(figure_save_path, 'full/'))
    print("Testing short")
    test(best_model, device, test_short_loader, os.path.join(figure_save_path, 'short/'))
    print("Testing long")
    test(best_model, device, test_long_loader, os.path.join(figure_save_path, 'long/'))