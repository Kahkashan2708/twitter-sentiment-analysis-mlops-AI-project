import os
import time
import joblib
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# ------------------ Project Setup ------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model.joblib")

# Load trained model
print(f"📂 Loading model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# Create FastAPI app
app = FastAPI(title="Twitter Sentiment Analysis API")

# Setup templates
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Ensure logs directory
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)
LOG_FILE = os.path.join(DATA_DIR, "predictions_log.csv")

if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["text", "prediction", "sentiment"]).to_csv(LOG_FILE, index=False)

# Label mapping
label_map = {0: "negative", 1: "neutral", 2: "positive"}
emoji_map = {"negative": "😡", "neutral": "😐", "positive": "😊"}

# ------------------ Prometheus Metrics ------------------
REQUEST_COUNT = Counter("request_count", "Total number of requests", ["endpoint"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency (s)", ["endpoint"])
PREDICTION_COUNT = Counter("prediction_count", "Number of predictions by sentiment", ["sentiment"])

@app.middleware("http")
async def add_metrics(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    REQUEST_COUNT.labels(request.url.path).inc()
    REQUEST_LATENCY.labels(request.url.path).observe(process_time)
    return response

@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Expose Prometheus metrics endpoint."""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ------------------ Routes ------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict_ui", response_class=HTMLResponse)
async def predict_ui(request: Request, text: str = Form(...)):
    text_series = pd.Series([text])
    prediction = model.predict(text_series)[0]
    sentiment = label_map.get(prediction, "unknown")
    emoji = emoji_map.get(sentiment, "❓")

    # --- Save result to CSV ---
    df = pd.DataFrame([[text, int(prediction), sentiment]],
                      columns=["text", "prediction", "sentiment"])
    df.to_csv(LOG_FILE, mode="a", header=False, index=False)

    # --- Update metrics ---
    PREDICTION_COUNT.labels(sentiment).inc()

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "text": text, "sentiment": sentiment, "emoji": emoji},
    )

@app.post("/predict")
def predict_api(input_data: dict):
    """API endpoint (JSON-based)."""
    text_series = pd.Series([input_data["text"]])
    prediction = model.predict(text_series)[0]
    sentiment = label_map.get(prediction, "unknown")

    # --- Save result to CSV ---
    df = pd.DataFrame([[input_data["text"], int(prediction), sentiment]],
                      columns=["text", "prediction", "sentiment"])
    df.to_csv(LOG_FILE, mode="a", header=False, index=False)

    # --- Update metrics ---
    PREDICTION_COUNT.labels(sentiment).inc()

    return {"text": input_data["text"], "predicted_label": int(prediction), "sentiment": sentiment}


# import os
# import joblib
# import pandas as pd
# from fastapi import FastAPI, Request, Form
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates

# # ------------------ Project Setup ------------------

# # Project root
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model.joblib")

# # Load trained model
# print(f"📂 Loading model from: {MODEL_PATH}")
# model = joblib.load(MODEL_PATH)

# # Create FastAPI app
# app = FastAPI(title="Twitter Sentiment Analysis API")

# # Setup templates
# TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")
# templates = Jinja2Templates(directory=TEMPLATES_DIR)

# # Ensure logs directory
# DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# os.makedirs(DATA_DIR, exist_ok=True)
# LOG_FILE = os.path.join(DATA_DIR, "predictions_log.csv")

# # Initialize log file if not exists
# if not os.path.exists(LOG_FILE):
#     pd.DataFrame(columns=["text", "prediction", "sentiment"]).to_csv(LOG_FILE, index=False)

# # Label mapping
# label_map = {0: "negative", 1: "neutral", 2: "positive"}
# emoji_map = {"negative": "😡", "neutral": "😐", "positive": "😊"}


# # ------------------ Routes ------------------

# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})


# @app.post("/predict_ui", response_class=HTMLResponse)
# async def predict_ui(request: Request, text: str = Form(...)):
#     text_series = pd.Series([text])
#     prediction = model.predict(text_series)[0]
#     sentiment = label_map.get(prediction, "unknown")
#     emoji = emoji_map.get(sentiment, "❓")

#     # --- Save result to CSV ---
#     df = pd.DataFrame([[text, int(prediction), sentiment]],
#                       columns=["text", "prediction", "sentiment"])
#     df.to_csv(LOG_FILE, mode="a", header=False, index=False)

#     return templates.TemplateResponse(
#         "index.html",
#         {
#             "request": request,
#             "text": text,
#             "sentiment": sentiment,
#             "emoji": emoji,
#         },
#     )


# @app.post("/predict")
# def predict_api(input_data: dict):
#     """API endpoint (JSON-based)."""
#     text_series = pd.Series([input_data["text"]])
#     prediction = model.predict(text_series)[0]
#     sentiment = label_map.get(prediction, "unknown")

#     # --- Save result to CSV ---
#     df = pd.DataFrame([[input_data["text"], int(prediction), sentiment]],
#                       columns=["text", "prediction", "sentiment"])
#     df.to_csv(LOG_FILE, mode="a", header=False, index=False)

#     return {
#         "text": input_data["text"],
#         "predicted_label": int(prediction),
#         "sentiment": sentiment,
#     }
