import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset 
import torch
import gc

from utils.data.helper import load


class VectorsDataset(Dataset):
    def __init__(self, vectors):
        self._vectors = np.asarray(vectors)

    def __len__(self):
        return int(self._vectors.shape[0])

    def __getitem__(self, index):
        row = self._vectors[index]
        return torch.as_tensor(row)


def split_df_by_ratio(df: pd.DataFrame, shuffle: bool = True, random_state: int = 42):
    train_df, temp_df = train_test_split(df, test_size=0.2, shuffle=shuffle, random_state=random_state)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, shuffle=shuffle, random_state=random_state)
    return train_df, val_df, test_df




