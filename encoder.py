import torch
import gc
from tqdm import tqdm
import os
import configs
from models.AutoEncoder import VectorAutoencoder, HybridLoss 
from utils.data.helper import loads
from utils.data.vector import load_vectors_from_input, split_vectors, load_vectors_splits_from_json

# --- CÁC THAM SỐ CÓ THỂ THAY ĐỔI ---
MODEL_SAVE_PATH = 'data/model/autoencoder.pt'
VECTORS_SPLIT_DIR = 'data/split_vectors'
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 20
PATHS = configs.Paths()
FILES = configs.Files()
DEVICE = FILES.get_device()

def train(model, device, train_loader, optimizer, loss_function, epoch):
    model.train()
    total_train_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epoch}")
    for batch_idx, vectors in progress_bar:
        vectors = vectors.to(device)
        
        reconstructed = model(vectors)
        loss = loss_function(reconstructed, vectors)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{total_train_loss/(batch_idx+1):.6f}"})

    return total_train_loss

def val(model, device, val_loader, loss_function, epoch):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Validating Epoch {epoch}")
        for batch_idx, vectors in progress_bar:
            vectors = vectors.to(device)
            reconstructed = model(vectors)
            loss = loss_function(reconstructed, vectors)
            total_val_loss += loss.item()
    
    return total_val_loss



def test(model, device, test_loader, loss_function):
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing")
        for batch_idx, vectors in progress_bar:
            vectors = vectors.to(device)
            reconstructed = model(vectors)
            loss = loss_function(reconstructed, vectors)
            total_test_loss += loss.item()
    
    print(f"Test Loss: {total_test_loss:.6f}")
    return total_test_loss


if __name__ == "__main__":
    input_dataset = loads(PATHS.input)
    print(f"Loaded {len(input_dataset)} samples from input")
    
    # Extract vectors from PyG Data
    vectors = load_vectors_from_input(input_dataset)
    
    # Split train/val/test and save JSON
    if os.path.exists(VECTORS_SPLIT_DIR) and os.path.exists(os.path.join(VECTORS_SPLIT_DIR, 'train.json')):
        train_ds, val_ds, test_ds = load_vectors_splits_from_json(VECTORS_SPLIT_DIR)
    else:
        train_ds, val_ds, test_ds = split_vectors(vectors, VECTORS_SPLIT_DIR, shuffle=True, random_state=42)
        del vectors
        gc.collect()
    
    # Create DataLoaders
    train_loader = train_ds.get_loader(BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = val_ds.get_loader(BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader = test_ds.get_loader(BATCH_SIZE, shuffle=False, drop_last=False)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    device = DEVICE
    print(f"Sử dụng thiết bị: {device}")
    model = VectorAutoencoder(input_dim=769, compressed_dim=101).to(device)
    loss_function = HybridLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # --- Begin training loop ---
    best_val_loss = float('inf')
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    print("\nBegin training AutoEncoder...")
    for epoch in range(1, NUM_EPOCHS + 1):
        total_train_loss = train(model, device, train_loader, optimizer, loss_function, epoch)
        total_val_loss = val(model, device, val_loader, loss_function, epoch)
        
        print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {total_train_loss:.6f} | Val Loss: {total_val_loss:.6f}")

        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)  # Save all model
            print(f"-> Val loss cải thiện. Đã lưu model tốt nhất vào '{MODEL_SAVE_PATH}'")

    print("\n--- Begin testing ---")
    # Load best model and test
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    test_loss = test(model, device, test_loader, loss_function)
    
    print("\nTraining finished!")
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Model saved at '{MODEL_SAVE_PATH}'")