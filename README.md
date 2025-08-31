# Twitter Sentiment Analysis — MLOps / AI Project

Simple end-to-end pipeline for Twitter sentiment analysis (ingest → preprocess → train). The pipeline reads a CSV of tweets, preprocesses text, trains a classifier and saves the trained model.

## Features
- CSV ingestion
- Text preprocessing
- Train / test split and model training
- Persist trained model to disk

## Repo layout
- src/ — pipeline code (ingest, preprocess, train, run_pipeline.py)
- data/
  - raw/ — input CSV (tweets_clean.csv)
  - processed/ — train/test CSVs after preprocessing
- models/ — saved model artifact (model.joblib)
- README.md

## Requirements
- Python 3.8+
- Install requirements (if present):
  - Windows:
    - conda create -n twitter python=3.10
    - actvate the env : conda activate twitter
    - pip install -r requirements.txt

## Usage
Ensure the raw CSV is at data/raw/tweets_clean.csv, then run the pipeline:

- Windows (PowerShell / CMD)
  - python src\run_pipeline.py

The script runs three steps: ingest, preprocess, train. Trained model is saved to models\model.joblib by default.

## Example run and results
Example pipeline output (captured from a successful run):

Test accuracy: 0.8122
              precision    recall  f1-score   support

           0       0.82      0.80      0.81    160000
           2       0.80      0.83      0.81    160000
    accuracy                           0.81    320000
    macro avg       0.81      0.81      0.81    320000  
    weighted avg    0.81      0.81      0.81    320000

Saved trained model to twitter-sentiment-analysis-mlops-AI-project\models\model.joblib  

Model saved at: twitter-sentiment-analysis-mlops-AI-project\models\model.joblib  
Test Accuracy: 0.8122

## Notes
- Adjust file paths in src/run_pipeline.py if your layout differs.
- Add or update requirements.txt with needed packages (scikit-learn, pandas, joblib, etc.).
