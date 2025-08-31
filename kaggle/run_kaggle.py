import os
import torch

import configs
from models.GGCN import GGCN
from models.LMGNN import BertGGCN
from run import train, validate, plot_validation_loss
from test import test
from utils.data.helper import loads as load_datasets
from utils.data.input import train_val_test_split as split_dataset


def run_kaggle_train(
    input_dir: str = '/kaggle/input/lm-train/LMTrain/data/input/',
    output_model_dir: str = '/kaggle/working/trained_models/',
    model_filename: str = 'bertggcn.pt',
    model_dir: str = '/kaggle/input/lm-train/LMTrain/data/model/',
    finetune_filename: str = 'graphcodebert_finetune.pt',
    hugging_path_filename: str = 'graphcodebert_finetune_hf',
    random_state: int = 42,
    figure_save_path: str = '/kaggle/working/figures/',
    batch_size: int = 12,
    epochs: int = 30,
    k: float = 0.6,
    mode_lm: bool = True
):
    """
    Kaggle-ready train flow using prebuilt input PKLs from input_dir.
    - No argparse, no config paths. Only uses configs for hyperparams.
    - Loads all PKLs via datamanager.loads and splits via datamanager.train_val_test_split.
    """
    os.makedirs(output_model_dir, exist_ok=True)
    os.makedirs(figure_save_path, exist_ok=True)

    # Load merged dataset using project utility
    input_dataset = load_datasets(input_dir)
    print(f"Loaded input dataset of size: {len(input_dataset)} from {input_dir}")

    # Split into InputDataset wrappers using project utility
    proc_config = configs.Process()
    shuffle = proc_config.shuffle
    train_ds, val_ds, test_ds, test_short_ds, test_long_ds = split_dataset(input_dataset, shuffle=shuffle, save_path=output_model_dir)

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
        finetune_file = os.path.join(model_dir, finetune_filename)
        hugging_path = os.path.join(output_model_dir, hugging_path_filename)
        model = BertGGCN(gated_graph_conv_args, conv_args, emb_size, device, k, hugging_path, finetune_file).to(device)
        best_model = BertGGCN(gated_graph_conv_args, conv_args, emb_size, device, k, hugging_path, finetune_file).to(device)
    else:
        model = GGCN(gated_graph_conv_args, conv_args, emb_size, device).to(device)
        best_model = GGCN(gated_graph_conv_args, conv_args, emb_size, device).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=bertggnn.learning_rate, weight_decay=bertggnn.weight_decay)

    # Train loop
    best_f1 = 0.0
    best_path = os.path.join(output_model_dir, model_filename)
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