import torch
import gc
from tqdm import tqdm
import os
import configs
from models.AutoEncoder import VectorAutoencoder, HybridLoss 
from utils.data.helper import loads
from utils.data.vector import split_df_by_ratio, VectorsDataset
from torch.utils.data import DataLoader, ConcatDataset
from utils.figure.plot import plot_validation_loss


# --- CÁC THAM SỐ CÓ THỂ THAY ĐỔI ---
MODEL_SAVE_PATH = 'data/model/autoencoder.pt'
INPUT_DIR = 'data/input'
FIGURE_SAVE_PATH = 'workspace/'
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 5
PATHS = configs.Paths()
FILES = configs.Files()
DEVICE = FILES.get_device()


def train(model, device, train_loader, optimizer, loss_function, epoch):
    model.train()
    total_train_loss = 0.0
    num_batches = 0
    progress_bar = tqdm(train_loader, total=len(train_loader), desc=f"Training Epoch {epoch}")
    for vectors in progress_bar:
        vectors = vectors.to(device)
        reconstructed = model(vectors)
        loss = loss_function(reconstructed, vectors)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        num_batches += 1
        avg = total_train_loss / max(num_batches, 1)
        progress_bar.set_postfix({"loss": f"{avg:.6f}"})
    
    return total_train_loss


def val(model, device, val_loader, loss_function, label, epoch = 0):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        progress_bar = tqdm(val_loader, total=len(val_loader), desc=f"{label} Epoch {epoch}")
        for vectors in progress_bar:
            vectors = vectors.to(device)
            reconstructed = model(vectors)
            loss = loss_function(reconstructed, vectors)
            total_loss += loss.item()
    return total_loss


if __name__ == "__main__":
    input_dataset = loads(INPUT_DIR)
    train_df, val_df, test_df = split_df_by_ratio(input_dataset, shuffle=True, random_state=42)

    train_sets = [VectorsDataset(obj.x.detach().cpu().numpy()) for obj in train_df['input']]
    val_sets = [VectorsDataset(obj.x.detach().cpu().numpy()) for obj in val_df['input']]
    test_sets = [VectorsDataset(obj.x.detach().cpu().numpy()) for obj in test_df['input']]

    train_concat = ConcatDataset(train_sets)
    val_concat = ConcatDataset(val_sets)
    test_concat = ConcatDataset(test_sets)

    train_loader = DataLoader(train_concat, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_concat, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_concat, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    device = DEVICE
    print(f"Using device: {device}")

    # Model, loss, optimizer
    model = VectorAutoencoder(input_dim=769, compressed_dim=101).to(device)
    loss_function = HybridLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_loss = float('inf')
    losses = []
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    print("\nBegin training AutoEncoder...")
    for epoch in range(1, NUM_EPOCHS + 1):
        total_train_loss = train(model, device, train_loader, optimizer, loss_function, epoch)
        total_val_loss = val(model, device, val_loader, loss_function, label="Val", epoch=epoch)
        losses.append(total_val_loss)
        print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {total_train_loss:.6f} | Val Loss: {total_val_loss:.6f}")

        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"-> Val loss cải thiện. Đã lưu model tốt nhất vào '{MODEL_SAVE_PATH}'")

    plot_validation_loss(losses, os.path.join(FIGURE_SAVE_PATH, 'validation_loss_autoencoder.png'))

    print("\n--- Begin testing ---")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    test_loss = val(model, device, test_loader, loss_function, label="Test", epoch=0)
    
    print("\nTraining finished!")
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Model saved at '{MODEL_SAVE_PATH}'")