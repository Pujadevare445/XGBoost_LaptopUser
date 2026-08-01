import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained XGBoost model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'XGBoost.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form submission
        age = float(request.form.get('age', 0))
        gender = int(request.form.get('gender', 0))
        region = int(request.form.get('region', 0))
        occupation = int(request.form.get('occupation', 0))
        income = float(request.form.get('income', 0))

        # Array shape: (1, 5) matching model requirement
        input_data = np.array([[age, gender, region, occupation, income]])

        # Get prediction and probability
        prediction = model.predict(input_data)[0]
        
        # Get probability if model supports it
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_data)[0][1]) * 100
        else:
            probability = None

        result_text = "Positive" if prediction == 1 else "Negative"

        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'result_text': result_text,
            'probability': round(probability, 2) if probability is not None else "N/A"
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
