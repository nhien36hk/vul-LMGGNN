import glob
import pickle

import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from ..functions.input_dataset import InputDataset
from ..functions import parse
from sklearn.model_selection import train_test_split


def read(path, json_file):
    """
    :param path: str
    :param json_file: str
    :return DataFrame
    """
    return pd.read_json(path + json_file)


def get_ratio(dataset, ratio):
    approx_size = int(len(dataset) * ratio)
    return dataset[:approx_size]


def load(path, pickle_file, ratio=1):
    dataset = pd.read_pickle(path + pickle_file)
    dataset.info(memory_usage='deep')
    if ratio < 1:
        dataset = get_ratio(dataset, ratio)

    return dataset


def write(data_frame: pd.DataFrame, path, file_name):
    data_frame.to_pickle(path + file_name)
    

def save_input_dataset(input_dataset, path, file_name):
    with open(path + file_name, 'wb') as f:
        pickle.dump(input_dataset, f)

def apply_filter(data_frame: pd.DataFrame, filter_func):
    return filter_func(data_frame)


def rename(data_frame: pd.DataFrame, old, new):
    return data_frame.rename(columns={old: new})


'''
def tokenize(data_frame: pd.DataFrame):
    data_frame.func = data_frame.func.apply(parse.tokenizer)
    # Change column name
    data_frame = rename(data_frame, 'func', 'tokens')
    # Keep just the tokens
    return data_frame[["tokens"]]
'''


def tokenize(data_frame: pd.DataFrame):
    data_frame["tokens"] = data_frame["func"].apply(parse.tokenizer)
    # Change column name
    # data_frame.rename(columns={"func": "tokens"}, inplace=True)
    # Keep just the tokens
    return data_frame[["tokens", "func"]]


def to_files(data_frame: pd.DataFrame, out_path):
    # path = f"{self.out_path}/{self.dataset_name}/"
    os.makedirs(out_path)

    for idx, row in data_frame.iterrows():
        file_name = f"{idx}.c"
        with open(out_path + file_name, 'w') as f:
            f.write(row.func)


def create_with_index(data, columns):
    data_frame = pd.DataFrame(data, columns=columns)
    data_frame.index = list(data_frame["Index"])

    return data_frame


def inner_join_by_index(df1, df2):
    return pd.merge(df1, df2, left_index=True, right_index=True)


def check_file_exists(file_path):
    return os.path.isfile(file_path)


def train_val_test_split(data_frame: pd.DataFrame, shuffle=True, save_path=None):
    print("Splitting Dataset")

    false = data_frame[data_frame.target == 0]
    true = data_frame[data_frame.target == 1]
    
    print(f"Total samples: {len(data_frame)}")
    print(f"Ratio False: {len(false) / len(data_frame)}")
    print(f"Ratio True: {len(true) / len(data_frame)}")

    # split false
    train_false, test_false = train_test_split(false, test_size=0.2, shuffle=shuffle)
    test_false, val_false = train_test_split(test_false, test_size=0.5, shuffle=shuffle)
    
    # split true
    train_true, test_true = train_test_split(true, test_size=0.2, shuffle=shuffle)
    test_true, val_true = train_test_split(test_true, test_size=0.5, shuffle=shuffle)

    # run = train_false.append(train_true)
    train = pd.concat([train_false, train_true])

    # val = val_false.append(val_true)
    val = pd.concat([val_false, val_true])

    # test = test_false.append(test_true)
    test = pd.concat([test_false, test_true])

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train_input = InputDataset(train)
    val_input = InputDataset(val)
    test_input = InputDataset(test)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        save_input_dataset(train_input, save_path, 'train.pkl')
        save_input_dataset(val_input, save_path, 'val.pkl')
        save_input_dataset(test_input, save_path, 'test.pkl')
        print(f"Saved split datasets to {save_path}")

    return train_input, val_input, test_input


def get_directory_files(directory):
    return [os.path.basename(file) for file in glob.glob(f"{directory}/*.pkl")]


def loads(data_sets_dir, ratio=1):
    data_sets_files = sorted([f for f in listdir(data_sets_dir) if isfile(join(data_sets_dir, f))])

    if ratio < 1:
        data_sets_files = get_ratio(data_sets_files, ratio)

    dataset = load(data_sets_dir, data_sets_files[0])
    data_sets_files.remove(data_sets_files[0])

    for ds_file in data_sets_files:
        # dataset = dataset.append(load(data_sets_dir, ds_file))
        dataset = pd.concat([dataset, load(data_sets_dir, ds_file)])

    return dataset


def clean(data_frame: pd.DataFrame):
    return data_frame.drop_duplicates(subset="func", keep=False)


def drop(data_frame: pd.DataFrame, keys):
    for key in keys:
        del data_frame[key]


def slice_frame(data_frame: pd.DataFrame, size: int):
    data_frame_size = len(data_frame)
    return data_frame.groupby(np.arange(data_frame_size) // size)


def check_split_exists(split_dir):
    train_file = os.path.join(split_dir, 'train.pkl')
    test_file = os.path.join(split_dir, 'test.pkl')
    val_file = os.path.join(split_dir, 'val.pkl')

    return os.path.isfile(train_file) and os.path.isfile(test_file) and os.path.isfile(val_file)


def load_input_dataset(path, file_name):
    with open(path + file_name, 'rb') as f:
        return pickle.load(f)


def load_split_datasets(path):
    train_dataset = load_input_dataset(path, 'train.pkl')
    val_dataset = load_input_dataset(path, 'val.pkl')
    test_dataset = load_input_dataset(path, 'test.pkl')
    return train_dataset, val_dataset, test_dataset
