import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

# Base directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

# Initialize Flask app
app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Load the trained XGBoost model safely
MODEL_PATH = os.path.join(BASE_DIR, 'XGBoost.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("XGBoost model loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Model file not found at {MODEL_PATH}")
    model = None
except Exception as e:
    print(f"ERROR loading model: {e}")
    model = None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model file is missing or failed to load on the server.'
        }), 500

    try:
        # Extract features from form submission
        age = float(request.form.get('age', 0))
        gender = int(request.form.get('gender', 0))
        region = int(request.form.get('region', 0))
        occupation = int(request.form.get('occupation', 0))
        income = float(request.form.get('income', 0))

        # Array shape: (1, 5) matching model input requirement
        input_data = np.array([[age, gender, region, occupation, income]])

        # Get prediction
        prediction = model.predict(input_data)[0]

        # Get probability if supported
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_data)[0][1]) * 100
        else:
            probability = None

        result_text = "Positive" if int(prediction) == 1 else "Negative"

        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'result_text': result_text,
            'probability': round(probability, 2) if probability is not None else "N/A"
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    # Binds dynamically to Render's PORT or defaults to 5000 locally
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
