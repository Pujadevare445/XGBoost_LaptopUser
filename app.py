import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the trained XGBoost model
try:
    with open('XGBoost.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Single-file HTML, CSS (Styling), and Interactive JS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XGBoost Model Deployment</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent-color: #6366f1;
            --accent-hover: #4f46e5;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: #334155;
            --input-bg: #0f172a;
            --success-color: #10b981;
            --danger-color: #ef4444;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 480px;
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 2.5rem;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
        }

        .header {
            margin-bottom: 2rem;
            text-align: center;
        }

        .header h1 {
            font-size: 1.75rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            margin-bottom: 0.5rem;
            background: linear-gradient(to right, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.875rem;
        }

        .form-group {
            margin-bottom: 1.25rem;
        }

        .form-group label {
            display: block;
            font-size: 0.875rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
            color: var(--text-main);
        }

        .form-group input, .form-group select {
            width: 100%;
            padding: 0.75rem 1rem;
            background-color: var(--input-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-main);
            font-size: 0.95rem;
            transition: all 0.2s ease-in-out;
            outline: none;
        }

        .form-group input:focus, .form-group select:focus {
            border-color: var(--accent-color);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
        }

        .btn-submit {
            width: 100%;
            padding: 0.875rem;
            background-color: var(--accent-color);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s ease-in-out, transform 0.1s ease;
            margin-top: 1rem;
        }

        .btn-submit:hover {
            background-color: var(--accent-hover);
        }

        .btn-submit:active {
            transform: scale(0.98);
        }

        #result-card {
            margin-top: 1.75rem;
            padding: 1.25rem;
            border-radius: 8px;
            display: none;
            text-align: center;
            border: 1px solid transparent;
            animation: fadeIn 0.3s ease-in-out forwards;
        }

        #result-card.success {
            background-color: rgba(16, 185, 129, 0.1);
            border-color: rgba(16, 185, 129, 0.3);
            color: var(--success-color);
        }

        #result-card.danger {
            background-color: rgba(239, 68, 68, 0.1);
            border-color: rgba(239, 68, 68, 0.3);
            color: var(--danger-color);
        }

        .result-title {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.25rem;
            opacity: 0.8;
        }

        .result-value {
            font-size: 1.25rem;
            font-weight: 700;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>XGBoost Predictor</h1>
        <p>Enter parameter values to generate a model prediction.</p>
    </div>

    <form id="prediction-form">
        <div class="form-group">
            <label for="age">Age</label>
            <input type="number" id="age" name="Age" placeholder="e.g. 35" required min="0" max="120">
        </div>

        <div class="form-group">
            <label for="gender">Gender</label>
            <select id="gender" name="Gender" required>
                <option value="" disabled selected>Select gender</option>
                <option value="0">Female (0)</option>
                <option value="1">Male (1)</option>
            </select>
        </div>

        <div class="form-group">
            <label for="region">Region</label>
            <input type="number" id="region" name="Region" placeholder="Region code (e.g. 0, 1, 2)" required>
        </div>

        <div class="form-group">
            <label for="occupation">Occupation</label>
            <input type="number" id="occupation" name="Occupation" placeholder="Occupation code (e.g. 0, 1, 2)" required>
        </div>

        <div class="form-group">
            <label for="income">Income</label>
            <input type="number" id="income" name="Income" placeholder="e.g. 50000" required step="any">
        </div>

        <button type="submit" class="btn-submit" id="submit-btn">Predict Outcome</button>
    </form>

    <div id="result-card">
        <div class="result-title">Prediction Result</div>
        <div class="result-value" id="result-text">--</div>
    </div>
</div>

<script>
    document.getElementById('prediction-form').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const btn = document.getElementById('submit-btn');
        const resultCard = document.getElementById('result-card');
        const resultText = document.getElementById('result-text');

        btn.textContent = 'Processing...';
        btn.disabled = true;

        const formData = {
            Age: parseFloat(document.getElementById('age').value),
            Gender: parseInt(document.getElementById('gender').value),
            Region: parseInt(document.getElementById('region').value),
            Occupation: parseInt(document.getElementById('occupation').value),
            Income: parseFloat(document.getElementById('income').value)
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            const data = await response.json();

            if (response.ok) {
                resultCard.className = 'success';
                resultText.textContent = `Class: ${data.prediction} (Probability: ${(data.probability * 100).toFixed(1)}%)`;
                resultCard.style.display = 'block';
            } else {
                resultCard.className = 'danger';
                resultText.textContent = data.error || 'An error occurred during prediction.';
                resultCard.style.display = 'block';
            }
        } catch (err) {
            resultCard.className = 'danger';
            resultText.textContent = 'Failed to connect to server.';
            resultCard.style.display = 'block';
        } finally {
            btn.textContent = 'Predict Outcome';
            btn.disabled = false;
        }
    });
</script>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model pickle file not loaded correctly on server.'}), 500

    try:
        data = request.get_json()
        
        # Format input feature vector matching training feature order:
        # [Age, Gender, Region, Occupation, Income]
        features = np.array([[
            data['Age'],
            data['Gender'],
            data['Region'],
            data['Occupation'],
            data['Income']
        ]])

        # Execute prediction
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        prob = float(probabilities[prediction])

        return jsonify({
            'prediction': prediction,
            'probability': prob
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
