import os
import pickle
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# Absolute path resolution ensures Render finds model.pkl relative to app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# Target model feature schema
EXPECTED_FEATURES = ["Age", "Gender", "Region", "Occupation", "Income"]

model = None


def load_model():
    """Safely loads the serialized model into memory on server launch."""
    global model
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            print("Successfully loaded model.pkl")
        except Exception as e:
            print(f"Error loading pickle file: {e}")
    else:
        print(f"Warning: '{MODEL_PATH}' not found. Verify it is committed to Git.")


# Initialize model on module load (required for Gunicorn on Render)
load_model()


@app.route("/", methods=["GET"])
def home():
    """
    Root Endpoint
    Prevents 'Not Found' / 404 errors when visiting the base URL in a browser.
    """
    return jsonify({
        "status": "online",
        "service": "XGBoost Classification API",
        "endpoints": {
            "health_check": "GET /health",
            "prediction": "POST /predict"
        }
    }), 200


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint to verify server status and model loading."""
    return jsonify({
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None
    }), 200 if model is not None else 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Inference Endpoint

    Accepts single JSON dict or a list of dicts:
    {
        "Age": 30,
        "Gender": 1,
        "Region": 2,
        "Occupation": 4,
        "Income": 55000
    }
    """
    if model is None:
        return jsonify({
            "status": "error",
            "message": "Model is not loaded. Ensure model.pkl exists in the project root."
        }), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({
            "status": "error",
            "message": "Invalid payload. Provide a valid JSON object."
        }), 400

    # Handle both single JSON objects and list of records
    is_single = isinstance(data, dict)
    records = [data] if is_single else data

    try:
        df = pd.DataFrame(records)

        # Strip invisible BOM characters if present
        df.columns = df.columns.str.replace("\ufeff", "")

        # Check missing features
        missing = [col for col in EXPECTED_FEATURES if col not in df.columns]
        if missing:
            return jsonify({
                "status": "error",
                "message": "Missing required input features",
                "missing_features": missing
            }), 400

        # Cast to numeric array matching expected feature order
        df_inference = df[EXPECTED_FEATURES].astype(int)

        # Perform inference
        predictions = model.predict(df_inference).tolist()
        probabilities = model.predict_proba(df_inference)[:, 1].tolist()

        results = [
            {"prediction": int(pred), "probability": float(prob)}
            for pred, prob in zip(predictions, probabilities)
        ]

        return jsonify({
            "status": "success",
            "data": results[0] if is_single else results
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Inference failed: {str(e)}"
        }), 500


if __name__ == "__main__":
    # Local execution fallback
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
