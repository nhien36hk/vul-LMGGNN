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
    

def save_split_indices(indices_map, path, file_name='split_idx.pkl'):
    with open(os.path.join(path, file_name), 'wb') as f:
        pickle.dump(indices_map, f)

def load_split_indices(path, file_name='split_idx.pkl'):
    with open(os.path.join(path, file_name), 'rb') as f:
        return pickle.load(f)

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


def count_tokens(code_text):
    return len(parse.tokenizer(code_text))


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


def split_long_short_test(test_true, test_false):
    tt = test_true
    tf = test_false

    # Compute token lengths
    tt = tt.assign(token_len=tt['func'].apply(count_tokens))
    tf = tf.assign(token_len=tf['func'].apply(count_tokens))

    # Compute Q25 and Q75 across the combined test distribution
    all_len = pd.concat([tt['token_len'], tf['token_len']])
    q25, q75 = all_len.quantile([0.25, 0.75])

    # Filter within each class to preserve class composition
    true_short = tt.loc[tt['token_len'] <= q25]
    true_long = tt.loc[tt['token_len'] >= q75]
    false_short = tf.loc[tf['token_len'] <= q25]
    false_long = tf.loc[tf['token_len'] >= q75]

    # Keep original indices; do not reset here
    test_short = pd.concat([true_short, false_short]).drop(columns=['token_len'])
    test_long = pd.concat([true_long, false_long]).drop(columns=['token_len'])

    print(f"Split thresholds (tokens): Q25={int(q25)}, Q75={int(q75)}")
    print(f"Test short: {len(test_short)}")
    print(f"Test long: {len(test_long)}")

    return test_short, test_long


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

    # Combine all splits (preserve original indices)
    train = pd.concat([train_false, train_true])
    val = pd.concat([val_false, val_true])
    test = pd.concat([test_false, test_true])

    # Compute short/long DataFrames
    test_short, test_long = split_long_short_test(test_true, test_false)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        indices_map = {
            'train': train.index.to_list(),
            'val': val.index.to_list(),
            'test': test.index.to_list(),
            'short': test_short.index.to_list(),
            'long': test_long.index.to_list()
        }
        save_split_indices(indices_map, save_path, 'split_idx.pkl')
        print(f"Saved split indices to {save_path}")

    # Wrap into InputDataset for runtime usage (in-memory)
    train_input = InputDataset(train)
    val_input = InputDataset(val)
    test_input = InputDataset(test)
    test_short_input = InputDataset(test_short)
    test_long_input = InputDataset(test_long)

    return train_input, val_input, test_input, test_short_input, test_long_input


def get_directory_files(directory):
    return [os.path.basename(file) for file in glob.glob(f"{directory}/*.pkl")]


def loads(data_sets_dir, ratio=1):
    data_sets_files = sorted([f for f in listdir(data_sets_dir) if isfile(join(data_sets_dir, f))])

    if ratio < 1:
        data_sets_files = get_ratio(data_sets_files, ratio)

    dataset = load(data_sets_dir, data_sets_files[0])
    data_sets_files.remove(data_sets_files[0])

    for ds_file in data_sets_files:
        dataset = pd.concat([dataset, load(data_sets_dir, ds_file)])

    dataset = dataset.reset_index(drop=True)
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
    return os.path.isfile(os.path.join(split_dir, 'split_idx.pkl'))


def load_split_datasets(path, dataset):
    indices = load_split_indices(path, 'split_idx.pkl')

    def by_idx(idxs):
        return dataset.loc[idxs].reset_index(drop=True)

    train_df = by_idx(indices['train'])
    val_df = by_idx(indices['val'])
    test_df = by_idx(indices['test'])
    test_short_df = by_idx(indices['short'])
    test_long_df = by_idx(indices['long'])

    return (InputDataset(train_df), InputDataset(val_df), InputDataset(test_df),
            InputDataset(test_short_df), InputDataset(test_long_df))
