import pickle
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load model from saved pickle binary
MODEL_PATH = "model.pkl"


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


try:
    model = load_model()
except Exception as e:
    model = None
    print(f"Error loading model: {e}")


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "status": "online",
            "message": "XGBoost Classification API is running.",
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded properly"}), 500

    try:
        data = request.get_json()

        # Extract features: Age, Gender, Region, Occupation, Income
        age = int(data.get("Age"))
        gender = int(data.get("Gender"))
        region = int(data.get("Region"))
        occupation = int(data.get("Occupation"))
        income = int(data.get("Income"))

        features = [[age, gender, region, occupation, income]]

        # Prediction & Probability
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]

        return jsonify(
            {
                "status": "success",
                "prediction": prediction,
                "confidence": float(probabilities[prediction]),
                "probabilities": {
                    "class_0": float(probabilities[0]),
                    "class_1": float(probabilities[1]),
                },
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
