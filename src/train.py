import joblib
import os
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
# from features import build_tfidf_pipeline
# from src.features import build_tfidf_pipeline
from src.features import build_tfidf_pipeline


def train_model(train_csv, test_csv, model_out="../model.joblib"):
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    tfidf = build_tfidf_pipeline()
    clf = LinearSVC(max_iter=10000)

    pipeline = Pipeline([("tfidf", tfidf), ("clf", clf)])
    param_grid = {"clf__C": [0.1, 1, 10]}

    gs = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=2, verbose=1)
    gs.fit(train['text'].fillna(""), train['label'])

    preds = gs.predict(test['text'].fillna(""))
    acc = accuracy_score(test['label'], preds)

    print(f" Test accuracy: {acc:.4f}")
    print(classification_report(test['label'], preds))

    joblib.dump(gs.best_estimator_, model_out)
    print(f" Saved trained model to {model_out}")

    return gs.best_estimator_, acc
