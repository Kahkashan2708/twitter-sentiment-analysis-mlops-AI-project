import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
# from features import build_tfidf_pipeline
from src.features import build_tfidf_pipeline

def run_preprocess(raw_csv, out_dir="../data/processed", test_size=0.2, random_state=42):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(raw_csv).dropna(subset=['text'])

    # split into train/test
    train, test = train_test_split(
        df, test_size=test_size, stratify=df['label'], random_state=random_state
    )

    train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(out_dir, "test.csv"), index=False)

    # build tfidf pipeline and fit on train text
    pipeline = build_tfidf_pipeline()
    pipeline.fit(train['text'].fillna(""))

    joblib.dump(pipeline, os.path.join(out_dir, "tfidf_pipeline.joblib"))
    print(" Preprocessing done: saved train/test and tfidf pipeline")
    return {
        "train_path": os.path.join(out_dir, "train.csv"),
        "test_path": os.path.join(out_dir, "test.csv"),
        "tfidf_path": os.path.join(out_dir, "tfidf_pipeline.joblib")
    }
