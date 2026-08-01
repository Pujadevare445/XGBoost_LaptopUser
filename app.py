import os
import pickle
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
EXPECTED_FEATURES = ["Age", "Gender", "Region", "Occupation", "Income"]

model = None

def load_model():
    """Loads model on startup."""
    global model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Model successfully loaded!")
    else:
        print(f"Warning: '{MODEL_PATH}' not found.")

@app.route("/", methods=["GET"])
def home():
    """Root route to test if API is live in a browser."""
    return jsonify({
        "status": "online",
        "message": "XGBoost Model API on Render",
        "endpoints": {
            "health": "/health",
            "predict": "POST /predict"
        }
    }), 200

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded."}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON payload"}), 400

    is_single_record = isinstance(data, dict)
    if is_single_record:
        data = [data]

    try:
        df = pd.DataFrame(data)
        df.columns = df.columns.str.replace("\ufeff", "")

        missing_cols = [c for c in EXPECTED_FEATURES if c not in df.columns]
        if missing_cols:
            return jsonify({
                "error": "Missing required features",
                "missing_columns": missing_cols
            }), 400

        df_inference = df[EXPECTED_FEATURES].astype(int)

        predictions = model.predict(df_inference).tolist()
        probabilities = model.predict_proba(df_inference)[:, 1].tolist()

        results = [
            {"prediction": int(pred), "probability": float(prob)}
            for pred, prob in zip(predictions, probabilities)
        ]

        output = results[0] if is_single_record else results
        return jsonify({"status": "success", "data": output}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Load model when app starts
load_model()

if __name__ == "__main__":
    # Render assigns its own port dynamically
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
