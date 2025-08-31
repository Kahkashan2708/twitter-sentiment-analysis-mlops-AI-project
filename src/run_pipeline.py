# src/run_pipeline.py
import os
from src.data_ingest import ingest_csv
from src.preprocess import run_preprocess
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    print(" Starting Twitter Sentiment Analysis Pipeline...")

    # 1. Ingest
    print("\n[Step 1] Ingesting dataset...")
    raw_path = ingest_csv(
        os.path.join("data", "raw", "tweets_clean.csv"),
        out_dir=os.path.join("data", "raw")
    )

    # 2. Preprocess
    print("\n[Step 2] Preprocessing dataset...")
    paths = run_preprocess(raw_path, out_dir=os.path.join("data", "processed"))

    # 3. Train
    print("\n[Step 3] Training model...")
    model, acc = train_model(
        paths["train_path"],
        paths["test_path"],
        model_out=os.path.join("models", "model.joblib")
    )

    print("\n Training finished")
    print(f"Model saved at: models/model.joblib")
    print(f"Test Accuracy: {acc:.4f}")

    # 4. Evaluate
    print("\n[Step 4] Evaluating model...")
    evaluate_model(
        model_path=os.path.join("models", "model.joblib"),
        test_path=paths["test_path"],
        report_dir=os.path.join("reports")
    )

    print("\n Pipeline finished successfully! Reports are saved in the 'reports/' folder.")

if __name__ == "__main__":
    print(">>> run_pipeline.py has started")
    main()
