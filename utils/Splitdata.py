import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(df: pd.DataFrame,text_columns: list, target_column: str, test_size=0.2, val_size=0.1, random_state=42):
    """
    Splits the dataset into training, validation, and testing sets.
    Returns:
        dict: A dictionary containing the train, validation, and test sets.
    """
    # Separate features and target
    X = df[text_columns]
    y = df[target_column]

    # First split: Train + Validation | Test for final test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: Train | Validation
    val_ratio = val_size / (1 - test_size)  
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=random_state, stratify=y_train_val
    )

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test)
    }