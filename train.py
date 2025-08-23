import os
import torch

import configs
from models.LMGNN import BertGGCN
from run import train, validate
from test import test
from utils.data.datamanager import loads as load_datasets, train_val_test_split as split_dataset


def run_kaggle_train(
    input_dir: str = '/kaggle/input/lm-train/LMTrain/data/input/',
    output_model_dir: str = '/kaggle/working/trained_models/',
    model_filename: str = 'bertggcn.pt',
    random_state: int = 42,
    figure_save_path: str = '/kaggle/working/figures/',
    batch_size: int = 64,
    k: float = 0.6,
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
    train_ds, test_ds, val_ds = split_dataset(input_dataset, shuffle=shuffle)

    # Build DataLoaders
    train_loader = train_ds.get_loader(batch_size, shuffle=True)
    val_loader = val_ds.get_loader(batch_size, shuffle=False)
    test_loader = test_ds.get_loader(batch_size, shuffle=False)

    # Model & Optimizer
    bertggnn = configs.BertGGNN()
    gated_graph_conv_args = bertggnn.model["gated_graph_conv_args"]
    conv_args = bertggnn.model["conv_args"]
    emb_size = bertggnn.model["emb_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertGGCN(gated_graph_conv_args, conv_args, emb_size, device, k).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=bertggnn.learning_rate, weight_decay=bertggnn.weight_decay)

    # Train loop
    best_acc = 0.0
    best_path = os.path.join(output_model_dir, model_filename)

    epochs = proc_config.epochs
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        acc, precision, recall, f1 = validate(model, device, val_loader, epoch)
        if best_acc < acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)
    print(f"Training finished. Best Acc: {best_acc:.4f}")

    # Load best and evaluate on test
    best_model = BertGGCN(gated_graph_conv_args, conv_args, emb_size, device, k).to(device)
    best_model.load_state_dict(torch.load(best_path, map_location=device))
    test(best_model, device, test_loader, figure_save_path)