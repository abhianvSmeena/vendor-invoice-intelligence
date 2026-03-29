import sqlite3

import pandas as pd
from sklearn.model_selection import train_test_split


def load_vendor_invoice_data(db_path: str) -> pd.DataFrame:
    """Load vendor invoice data from a SQLite database."""
    conn = sqlite3.connect(db_path)
    try:
        query = "SELECT * FROM vendor_invoice"
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


def prepare_features(df: pd.DataFrame):
    """Select model features and target variable."""
    X = df[["Dollars"]]
    y = df["Freight"]
    return X, y


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Split dataset into train and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
