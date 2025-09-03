import os
import torch

from encoder import train, val
from models.AutoEncoder import VectorAutoencoder, HybridLoss
from utils.data.helper import loads
from utils.data.vector import split_df_by_ratio, VectorsDataset
from torch.utils.data import DataLoader, ConcatDataset
from utils.figure.plot import plot_validation_loss


def run_kaggle_encoder(
    input_dir: str = '/kaggle/input/lm-train/LMTrain/data/input/',
    output_model_dir: str = '/kaggle/working/trained_models/',
    input_dim: int = 769,
    compressed_dim: int = 101,
    batch_size: int = 64,
    epochs: int = 20,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
):
    os.makedirs(output_model_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load input dataset directly (same as encoder.py)
    input_dataset = loads(input_dir)
    print(f"Loaded input DataFrame: {len(input_dataset)} rows")
    train_df, val_df, test_df = split_df_by_ratio(input_dataset, shuffle=True, random_state=42)

    # Create VectorsDataset from each function's x and concatenate
    train_sets = [VectorsDataset(obj.x.detach().cpu().numpy()) for obj in train_df['input']]
    val_sets = [VectorsDataset(obj.x.detach().cpu().numpy()) for obj in val_df['input']]
    test_sets = [VectorsDataset(obj.x.detach().cpu().numpy()) for obj in test_df['input']]

    # Concatenate vectors dataset
    train_concat = ConcatDataset(train_sets)
    val_concat = ConcatDataset(val_sets)
    test_concat = ConcatDataset(test_sets)

    # Create DataLoaders
    train_loader = DataLoader(train_concat, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_concat, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_concat, batch_size=batch_size, shuffle=False, drop_last=False)

    # Model, loss, optimizer
    model = VectorAutoencoder(input_dim=input_dim, compressed_dim=compressed_dim).to(device)
    loss_function = HybridLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float('inf')
    losses = []
    best_path = os.path.join(output_model_dir, 'autoencoder.pt')

    print("\nBegin training AutoEncoder (Kaggle)...")
    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, loss_function, epoch)
        val_loss = val(model, device, val_loader, loss_function, label="Val", epoch=epoch)
        losses.append(val_loss)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"-> Improved val loss. Saved best model to '{best_path}'")

    plot_validation_loss(losses, os.path.join(output_model_dir, 'validation_loss_autoencoder.png'))

    print("\n--- Begin testing (best model) ---")
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss = val(model, device, test_loader, loss_function, label="Test", epoch=0)

    print("\nTraining finished!")
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Model saved at '{best_path}'")


