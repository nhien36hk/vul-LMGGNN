import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader 
import torch
import gc

from utils.data.helper import load


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


def save_vectors_npz(vectors: np.ndarray, file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.savez_compressed(file_path, vectors=vectors.astype(np.float32))


def load_vectors_npz(file_path: str):
    return np.load(file_path)['vectors'].astype(np.float32)


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

    # save_vectors_npz(train_vectors, os.path.join(save_dir, 'train.npz'))
    # save_vectors_npz(val_vectors, os.path.join(save_dir, 'val.npz'))
    # save_vectors_npz(test_vectors, os.path.join(save_dir, 'test.npz'))

    return VectorsDataset(train_vectors), VectorsDataset(val_vectors), VectorsDataset(test_vectors)


def load_vectors_splits_from_npz(save_dir: str):
    train_vectors = load_vectors_npz(os.path.join(save_dir, 'train.npz'))
    val_vectors = load_vectors_npz(os.path.join(save_dir, 'val.npz'))
    test_vectors = load_vectors_npz(os.path.join(save_dir, 'test.npz'))
    return VectorsDataset(train_vectors), VectorsDataset(val_vectors), VectorsDataset(test_vectors)


def save_vector(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    for file_name in sorted(os.listdir(input_dir)):
        if not file_name.endswith('.pkl'):
            continue
        file_id = file_name.split('_')[0]
        out_name = f"{file_id}_vector.npz"
        out_path = os.path.join(output_dir, out_name)
        if os.path.exists(out_path):
            print(f"Vector đã tồn tại: {out_path}")
            continue

        df = load(input_dir, file_name)
        vectors = load_vectors_from_input(df)
        save_vectors_npz(vectors, out_path)

        del df
        del vectors
        gc.collect()


def load_vector_all_from_npz(vector_dir: str):
    vectors_all = []
    for file_name in sorted(os.listdir(vector_dir)):
        if not file_name.endswith('.npz'):
            continue
        in_path = os.path.join(vector_dir, file_name)
        vectors_all.extend(load_vectors_npz(in_path))
    vectors_all = np.asarray(vectors_all, dtype=np.float32)
    return vectors_all
