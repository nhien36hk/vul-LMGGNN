import os
import torch

from encoder import train, val, test
from models.AutoEncoder import VectorAutoencoder, HybridLoss
from utils.data.vector import load_vector_all_from_npz, split_vectors


def run_kaggle_encoder(
    vectors_dir: str = '/kaggle/input/lm-train/LMTrain/data/vector/',
    output_model_dir: str = '/kaggle/working/trained_models/',
    model_filename: str = 'autoencoder.pt',
    batch_size: int = 128,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    epochs: int = 20,
    input_dim: int = 769,
    compressed_dim: int = 101,
):
    os.makedirs(output_model_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vectors = load_vector_all_from_npz(vectors_dir)

    # Split to train/val/test and wrap in torch Dataset
    train_ds, val_ds, test_ds = split_vectors(vectors, save_dir='/kaggle/working/split_vector', shuffle=True, random_state=42)

    # Dataloaders
    train_loader = train_ds.get_loader(batch_size, shuffle=True, drop_last=True)
    val_loader = val_ds.get_loader(batch_size, shuffle=False, drop_last=False)
    test_loader = test_ds.get_loader(batch_size, shuffle=False, drop_last=False)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Model, loss, optimizer
    model = VectorAutoencoder(input_dim=input_dim, compressed_dim=compressed_dim).to(device)
    loss_function = HybridLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float('inf')
    best_path = os.path.join(output_model_dir, model_filename)

    print("\nBegin training AutoEncoder (Kaggle)...")
    for epoch in range(1, epochs + 1):
        total_train_loss = train(model, device, train_loader, optimizer, loss_function, epoch)
        total_val_loss = val(model, device, val_loader, loss_function, epoch)

        print(f"Epoch {epoch}/{epochs} | Train Loss: {total_train_loss:.6f} | Val Loss: {total_val_loss:.6f}")

        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            torch.save(model.state_dict(), best_path)
            print(f"-> Improved val loss. Saved best model to '{best_path}'")

    print("\n--- Begin testing (best model) ---")
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss = test(model, device, test_loader, loss_function)
    print(f"Test Loss: {test_loss:.6f}")

    print("\nTraining finished!")
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Model saved at '{best_path}'")


