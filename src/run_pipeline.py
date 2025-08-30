# src/run_pipeline.py
import os
from src.data_ingest import ingest_csv
from src.preprocess import run_preprocess
from src.train import train_model

# get project root automatically (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    print(">>> run_pipeline.py has started")
    print(" Starting Twitter Sentiment Analysis Pipeline...")

    # Paths
    raw_input = os.path.join(PROJECT_ROOT, "data", "raw", "tweets_clean.csv")
    raw_out   = os.path.join(PROJECT_ROOT, "data", "raw")
    processed_out = os.path.join(PROJECT_ROOT, "data", "processed")
    model_out = os.path.join(PROJECT_ROOT, "models", "model.joblib")

    # 1. Ingest
    print("\n[Step 1] Ingesting dataset...")
    raw_path = ingest_csv(raw_input, out_dir=raw_out)

    # 2. Preprocess
    print("\n[Step 2] Preprocessing dataset...")
    paths = run_preprocess(raw_path, out_dir=processed_out)

    # 3. Train
    print("\n[Step 3] Training model...")
    model, acc = train_model(paths["train_path"], paths["test_path"], model_out=model_out)

    print("\n Pipeline finished successfully!")
    print(f"Model saved at: {model_out}")
    print(f"Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
