import os
import pandas as pd

SOURCE_PATH = "data/creditcard.csv" if os.path.exists("data/creditcard.csv") else "creditcard.csv"


def split_dataset():
    if not os.path.exists(SOURCE_PATH):
        print(f"[Error] Khong tim thay tep: {SOURCE_PATH}")
        return

    df = pd.read_csv(SOURCE_PATH)
    split_idx = len(df) // 2
    os.makedirs("data", exist_ok=True)

    df.iloc[:split_idx].to_csv("data/creditcard_prefix.csv", index=False)
    df.iloc[split_idx:].to_csv("data/creditcard_suffix.csv", index=False)
    print(f"[Done] Phan tach du lieu: {split_idx} tien huan luyen, {len(df) - split_idx} luong danh gia.")


if __name__ == "__main__":
    split_dataset()
