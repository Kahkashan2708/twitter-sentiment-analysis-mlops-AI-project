# src/data_ingest.py
import pandas as pd
import os

def ingest_csv(input_path, out_dir):
    df = pd.read_csv(input_path)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "tweets_raw.csv")
    df.to_csv(out_path, index=False)

    print(f" Saved raw dataset to {out_path} (rows: {len(df)})")
    return out_path
