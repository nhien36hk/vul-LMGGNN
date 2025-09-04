import pandas as pd
import os
from sklearn.model_selection import train_test_split
from .helper import count_tokens, save_split_indices, load_split_indices
from ..functions.input_dataset import InputDataset


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