import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader 
import torch


class VectorsDataset(Dataset):
    def __init__(self, vectors):
        self._vectors = np.asarray(vectors)

    def __len__(self):
        return int(self._vectors.shape[0])

    def __getitem__(self, index):
        vec = self._vectors[index]
        if torch is not None:
            return torch.as_tensor(vec, dtype=torch.float32)
        return vec

    def get_loader(self, batch_size: int, shuffle: bool = True, drop_last: bool = False):
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def ensure_numpy_2d(vectors):
    arr = np.asarray(vectors)
    if arr.ndim != 2:
        raise ValueError(f"vectors must be 2D array-like, got shape {arr.shape}")
    return arr


def save_vectors_json(vectors: np.ndarray, file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(vectors.tolist(), f)


def load_vectors_json(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return np.asarray(data, dtype=np.float32)


def load_vectors_from_input(df) -> np.ndarray:
    if 'input' not in df.columns:
        raise ValueError("DataFrame phải có cột 'input' chứa PyG Data.")

    gathered = []
    for obj in df['input'].to_list():
        x = obj.x.detach().cpu().numpy()  # [num_nodes, D]
        gathered.extend(list(x))
    arr = np.asarray(gathered, dtype=np.float32)  # [TotalNodes, D]
    return arr


def split_vectors(vectors, save_dir: str,
                  shuffle: bool = True, random_state: int = 42):
    arr = ensure_numpy_2d(vectors)
    num_samples = arr.shape[0]
    print(f"Total samples: {num_samples}")

    indices = np.arange(num_samples)

    # 80% train, 20% temp
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, shuffle=shuffle, random_state=random_state)
    # 10% val, 10% test from temp
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, shuffle=shuffle, random_state=random_state)

    train_vectors = arr[train_idx]
    val_vectors = arr[val_idx]
    test_vectors = arr[test_idx]

    # save_vectors_json(train_vectors, os.path.join(save_dir, 'train.json'))
    # save_vectors_json(val_vectors, os.path.join(save_dir, 'val.json'))
    # save_vectors_json(test_vectors, os.path.join(save_dir, 'test.json'))

    return VectorsDataset(train_vectors), VectorsDataset(val_vectors), VectorsDataset(test_vectors)


def load_vectors_splits_from_json(save_dir: str):
    train_vectors = load_vectors_json(os.path.join(save_dir, 'train.json'))
    val_vectors = load_vectors_json(os.path.join(save_dir, 'val.json'))
    test_vectors = load_vectors_json(os.path.join(save_dir, 'test.json'))
    return VectorsDataset(train_vectors), VectorsDataset(val_vectors), VectorsDataset(test_vectors)
