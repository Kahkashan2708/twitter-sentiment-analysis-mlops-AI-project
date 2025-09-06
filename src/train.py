import joblib
import os
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from src.features import build_tfidf_pipeline

# MLflow
import mlflow
import mlflow.sklearn


def train_model(train_csv, test_csv, model_out="../models/model.joblib"):
    # --- Load Data ---
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    # --- Build Pipeline ---
    tfidf = build_tfidf_pipeline()
    clf = LinearSVC(max_iter=10000)
    pipeline = Pipeline([("tfidf", tfidf), ("clf", clf)])

    # --- Hyperparameter Search ---
    param_grid = {"clf__C": [0.1, 1, 10]}
    gs = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=2, verbose=1)

    # --- Training ---
    gs.fit(train['text'].fillna(""), train['label'])

    # --- Predictions ---
    preds = gs.predict(test['text'].fillna(""))
    acc = accuracy_score(test['label'], preds)
    precision = precision_score(test['label'], preds, average="weighted", zero_division=0)
    recall = recall_score(test['label'], preds, average="weighted", zero_division=0)
    f1 = f1_score(test['label'], preds, average="weighted", zero_division=0)

    print(f"\n Test accuracy: {acc:.4f}")
    print(classification_report(test['label'], preds))

    # --- Save Best Model Locally ---
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(gs.best_estimator_, model_out)
    print(f" Saved trained model to {model_out}")

    # --- MLflow Logging ---
    with mlflow.start_run():
        mlflow.log_param("model_type", "LinearSVC + TF-IDF")
        mlflow.log_params(gs.best_params_)  # logs best C value
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log model in MLflow
        mlflow.sklearn.log_model(gs.best_estimator_, "model")

    return gs.best_estimator_, acc

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python -m src.train <train_csv> <test_csv> <model_out>")
        sys.exit(1)

    train_csv = sys.argv[1]
    test_csv = sys.argv[2]
    model_out = sys.argv[3]

    os.makedirs(os.path.dirname(model_out), exist_ok=True)  # ensure models/ exists
    train_model(train_csv, test_csv, model_out)


# import joblib
# import os
# import pandas as pd
# from sklearn.svm import LinearSVC
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.model_selection import GridSearchCV
# # from features import build_tfidf_pipeline
# # from src.features import build_tfidf_pipeline
# from src.features import build_tfidf_pipeline


# def train_model(train_csv, test_csv, model_out="../model.joblib"):
#     train = pd.read_csv(train_csv)
#     test = pd.read_csv(test_csv)

#     tfidf = build_tfidf_pipeline()
#     clf = LinearSVC(max_iter=10000)

#     pipeline = Pipeline([("tfidf", tfidf), ("clf", clf)])
#     param_grid = {"clf__C": [0.1, 1, 10]}

#     gs = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=2, verbose=1)
#     gs.fit(train['text'].fillna(""), train['label'])

#     preds = gs.predict(test['text'].fillna(""))
#     acc = accuracy_score(test['label'], preds)

#     print(f" Test accuracy: {acc:.4f}")
#     print(classification_report(test['label'], preds))

#     joblib.dump(gs.best_estimator_, model_out)
#     print(f" Saved trained model to {model_out}")

#     return gs.best_estimator_, acc
