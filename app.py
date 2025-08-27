import os
import re
import json
import datetime as dt
from pathlib import Path
from typing import List, Dict, Any

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import joblib

from utils import clean_text, lemmatize_basic, explain_top_terms

# ------------------------------
# Config
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "uploads"
LOG_DIR = BASE_DIR / "logs"
UPLOAD_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "predictions.csv"

ALLOWED_EXT = {"csv", "txt"}

# ------------------------------
# Load models
# ------------------------------
rf = joblib.load(MODEL_DIR / "rf_model.pkl")
xgb = joblib.load(MODEL_DIR / "xgb_model.pkl")
lr = joblib.load(MODEL_DIR / "lr_model.pkl")

# lgbm may be None
lgbm_best = None
try:
    lgbm_best = joblib.load(MODEL_DIR / "lgbm_model.pkl")
except:
    print("⚠️ LGBM model not found, continuing without it.")

vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")

# optional explainer
LR_EXPLAINER = None
try:
    LR_EXPLAINER = joblib.load(MODEL_DIR / "logit_explainer.pkl")
except:
    pass

CLASS_NAMES = list(label_encoder.classes_)

# ------------------------------
# App
# ------------------------------
app = Flask(__name__)

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def preprocess_for_model(texts: List[str]) -> Any:
    """Clean + lemmatize (lightweight) + vectorize."""
    cleaned = [lemmatize_basic(clean_text(t)) for t in texts]
    return vectorizer.transform(cleaned), cleaned

def log_predictions(rows: List[Dict[str, Any]]):
    df = pd.DataFrame(rows)
    if LOG_FILE.exists():
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)

def get_ensemble_proba(X):
    """Average probabilities from available models"""
    probas = []
    for model in [rf, xgb, lgbm_best, lr]:
        if model is not None:
            try:
                probas.append(model.predict_proba(X))
            except Exception:
                preds = model.predict(X)
                arr = np.zeros((len(X), len(CLASS_NAMES)))
                for i, p in enumerate(preds):
                    arr[i, p] = 1.0
                probas.append(arr)
    return np.mean(probas, axis=0)

# ------------------------------
# Routes
# ------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", class_names=CLASS_NAMES)

@app.route("/predict", methods=["POST"])
def predict_form():
    text = request.form.get("text", "").strip()
    if not text:
        return render_template("index.html", error="Please enter some text.", class_names=CLASS_NAMES)

    X, cleaned = preprocess_for_model([text])
    y_proba = get_ensemble_proba(X)[0]

    y_idx = int(np.argmax(y_proba))
    label = CLASS_NAMES[y_idx]
    confidence = float(y_proba[y_idx])

    # explanation (top terms)
    top_terms = explain_top_terms(
        text=text,
        cleaned=cleaned[0],
        vectorizer=vectorizer,
        y_index=y_idx,
        class_names=CLASS_NAMES,
        lr_explainer=LR_EXPLAINER
    )

    log_predictions([{
        "timestamp": dt.datetime.utcnow().isoformat(),
        "text": text,
        "pred_label": label,
        "confidence": confidence
    }])

    proba_payload = [{"label": CLASS_NAMES[i], "value": float(p)} for i, p in enumerate(y_proba)]

    return render_template(
        "index.html",
        input_text=text,
        pred_label=label,
        confidence=f"{confidence*100:.2f}%",
        proba_json=json.dumps(proba_payload),
        top_terms=top_terms,
        class_names=CLASS_NAMES
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    texts = data.get("texts") or []
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({"error": "Send JSON with 'texts': [..]"}), 400

    X, cleaned = preprocess_for_model(texts)
    probs = get_ensemble_proba(X)

    preds_idx = probs.argmax(axis=1)
    labels = label_encoder.inverse_transform(preds_idx)

    rows_to_log = []
    results = []
    for i, t in enumerate(texts):
        label = labels[i]
        conf = float(probs[i, preds_idx[i]])
        rows_to_log.append({
            "timestamp": dt.datetime.utcnow().isoformat(),
            "text": t,
            "pred_label": label,
            "confidence": conf
        })
        top_terms = explain_top_terms(
            text=t,
            cleaned=cleaned[i],
            vectorizer=vectorizer,
            y_index=int(preds_idx[i]),
            class_names=CLASS_NAMES,
            lr_explainer=LR_EXPLAINER
        )
        results.append({
            "text": t,
            "label": label,
            "confidence": conf,
            "probs": {CLASS_NAMES[j]: float(probs[i, j]) for j in range(len(CLASS_NAMES))},
            "top_terms": top_terms
        })

    log_predictions(rows_to_log)
    return jsonify({"results": results})

@app.route("/batch", methods=["POST"])
def batch_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Only csv/txt allowed"}), 400

    fname = secure_filename(file.filename)
    path = UPLOAD_DIR / fname
    file.save(path)

    df = pd.read_csv(path) if fname.lower().endswith(".csv") else pd.read_table(path, header=None, names=["text"])
    if "text" not in df.columns:
        return jsonify({"error": "CSV must have a 'text' column"}), 400

    X, cleaned = preprocess_for_model(df["text"].astype(str).tolist())
    probs = get_ensemble_proba(X)

    preds_idx = probs.argmax(axis=1)
    labels = label_encoder.inverse_transform(preds_idx)
    confs = probs.max(axis=1)

    out = df.copy()
    out["pred_label"] = labels
    out["confidence"] = confs

    rows = [{
        "timestamp": dt.datetime.utcnow().isoformat(),
        "text": t,
        "pred_label": l,
        "confidence": float(c)
    } for t, l, c in zip(df["text"], labels, confs)]
    log_predictions(rows)

    out_name = f"predictions_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    out_path = UPLOAD_DIR / out_name
    out.to_csv(out_path, index=False)

    return jsonify({"message": "Batch processed", "download": f"/download/{out_name}", "rows": len(out)})

@app.route("/download/<path:filename>", methods=["GET"])
def download(filename):
    return send_from_directory(UPLOAD_DIR, filename, as_attachment=True)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "classes": CLASS_NAMES})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
