import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the pretrained XGBoost model
MODEL_PATH = "XGBoost.pkl"

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found in root directory.")

model = load_model()

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract features from form input
        age = float(request.form.get("Age", 0))
        gender = float(request.form.get("Gender", 0))
        region = float(request.form.get("Region", 0))
        occupation = float(request.form.get("Occupation", 0))
        income = float(request.form.get("Income", 0))

        # Format input for prediction
        input_data = np.array([[age, gender, region, occupation, income]])
        
        # Predict class and probability
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        confidence = float(probabilities[prediction]) * 100

        result_text = "Positive Result (Class 1)" if int(prediction) == 1 else "Negative Result (Class 0)"

        return render_template(
            "index.html",
            prediction_text=result_text,
            confidence_score=f"{confidence:.2f}%",
            status="success",
            inputs=request.form
        )

    except Exception as e:
        return render_template(
            "index.html",
            error_text=f"Error in prediction: {str(e)}",
            status="danger"
        )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
