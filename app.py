import os
import pickle
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# Model configuration and loading
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
EXPECTED_FEATURES = ["Age", "Gender", "Region", "Occupation", "Income"]

model = None


def load_model():
    """Load binary model file safely upon application startup."""
    global model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Model successfully loaded from disk.")
    else:
        print(f"Warning: Model file not found at '{MODEL_PATH}'. "
              "Inference requests will fail until present.")


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction Endpoint

    Expects JSON input:
    {
        "Age": 30,
        "Gender": 1,
        "Region": 2,
        "Occupation": 4,
        "Income": 55000
    }

    Or a list of dicts for batch predictions.
    """
    if model is None:
        return jsonify({"error": "Model is not loaded on server."}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON payload"}), 400

    # Ensure single item is handled gracefully alongside lists
    is_single_record = False
    if isinstance(data, dict):
        data = [data]
        is_single_record = True

    try:
        # Convert incoming payload to pandas DataFrame
        df = pd.DataFrame(data)

        # Handle feature validation (stripping BOM chars if present in raw metadata)
        df.columns = df.columns.str.replace("\ufeff", "")

        # Missing column check
        missing_cols = [col for col in EXPECTED_FEATURES if col not in df.columns]
        if missing_cols:
            return jsonify({
                "error": "Missing required feature columns",
                "missing_columns": missing_cols
            }), 400

        # Filter and reorder columns matching trained XGBoost sequence
        df_inference = df[EXPECTED_FEATURES].astype(int)

        # Perform predictions
        predictions = model.predict(df_inference).tolist()
        probabilities = model.predict_proba(df_inference)[:, 1].tolist()

        # Format output
        results = [
            {"prediction": int(pred), "probability": float(prob)}
            for pred, prob in zip(predictions, probabilities)
        ]

        output = results[0] if is_single_record else results
        return jsonify({"status": "success", "data": output}), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500


if __name__ == "__main__":
    load_model()
    # Run server
    app.run(host="0.0.0.0", port=5000, debug=True)
