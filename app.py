from flask import Flask, request, jsonify, render_template
import pickle
import xgboost as xgb
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model when the application starts
model = None
try:
    with open('waiting_time_model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

FEATURE_NAMES = [
    'number_of_items',
    'kitchen_current_load',
    'chefs_available',
    'load_per_chef',
    'item_x_load',
    'order_time_Lunch',
    'order_time_Morning',
    'priority_VIP',
    'order_size_medium',
    'order_size_small'
]

@app.route('/')
def home():
    return render_template('index.html')  # Serve from templates folder

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json(force=True)
        
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
            
        input_features = []
        for feature in FEATURE_NAMES:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
            input_features.append(data[feature])
        
        input_array = np.array(input_features).reshape(1, -1)
        prediction = model.predict(input_array)
        
        return jsonify({
            'predicted_waiting_time': float(prediction[0]),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    if model is not None:
        return jsonify({'status': 'healthy', 'model_loaded': True})
    else:
        return jsonify({'status': 'unhealthy', 'model_loaded': False}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)