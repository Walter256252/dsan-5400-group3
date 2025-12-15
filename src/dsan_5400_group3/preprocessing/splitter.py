"""
splitter.py

Helper utilities for train/val/test splitting (80/10/10).
"""

import pandas as pd
from pathlib import Path


def train_val_test_split(df: pd.DataFrame, random_state=42):
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    n = len(df)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val

    df_train = df.iloc[:n_train].copy()
    df_val = df.iloc[n_train:n_train + n_val].copy()
    df_test = df.iloc[n_train + n_val:].copy()

    return df_train, df_val, df_test
