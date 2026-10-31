"""
STEP 2: Load and inspect the Student Performance dataset.
"""
import pandas as pd
import os

DATA_PATH = "data/student-mat.csv"


def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path, sep=";")

    print("=" * 60)
    print("📊 DATASET LOADED SUCCESSFULLY")
    print("=" * 60)
    print(f"Shape       : {df.shape}")
    print(f"Rows        : {df.shape[0]}")
    print(f"Columns     : {df.shape[1]}")
    print(f"\nColumn Names:\n{df.columns.tolist()}")
    print(f"\nFirst 5 Rows:\n{df.head()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print("=" * 60)
    return df


if __name__ == "__main__":
    load_data()