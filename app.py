from flask import Flask, request, jsonify, render_template
import pickle
import xgboost as xgb
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Constants
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
    'order_size_small'  # Fixed: make sure this matches frontend
]

# Model loading with better validation
model = None
try:
    with open('waiting_time_model.pkl', 'rb') as f:
        model = pickle.load(f)
        # Verify the loaded model has predict method
        if not hasattr(model, 'predict'):
            raise AttributeError("Loaded model doesn't have predict method")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Early return if model not loaded
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'status': 'error'
        }), 500
    
    try:
        # Get and validate input data
        data = request.get_json(force=True)
        
        if not data or not isinstance(data, dict):
            return jsonify({
                'error': 'Invalid input format - expected JSON object',
                'status': 'error'
            }), 400
            
        # Validate all required fields exist
        missing_fields = [f for f in FEATURE_NAMES if f not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}',
                'status': 'error',
                'missing_fields': missing_fields
            }), 400
            
        # Type conversion and validation
        input_features = []
        for feature in FEATURE_NAMES:
            try:
                value = data[feature]
                
                # Handle different feature types
                if feature in ['order_time_Lunch', 'order_time_Morning', 
                             'priority_VIP', 'order_size_medium', 'order_size_small']:
                    # Boolean/flag features (should be 0 or 1)
                    val = int(value)
                    if val not in (0, 1):
                        raise ValueError(f"{feature} must be 0 or 1")
                else:
                    # Numeric features
                    val = float(value)
                    # Add additional validation if needed
                    if feature == 'number_of_items' and val < 1:
                        raise ValueError("Number of items must be at least 1")
                
                input_features.append(val)
                
            except ValueError as ve:
                return jsonify({
                    'error': f'Invalid value for {feature}: {str(ve)}',
                    'status': 'error'
                }), 400
            except Exception as e:
                return jsonify({
                    'error': f'Error processing {feature}: {str(e)}',
                    'status': 'error'
                }), 400
        
        # Make prediction
        input_array = np.array(input_features).reshape(1, -1)
        prediction = model.predict(input_array)
        
        # Ensure prediction is a single numeric value
        try:
            predicted_value = float(prediction[0])
        except (IndexError, TypeError, ValueError) as e:
            return jsonify({
                'error': f'Invalid prediction result: {str(e)}',
                'status': 'error'
            }), 500
        
        return jsonify({
            'predicted_waiting_time': predicted_value,
            'status': 'success',
            'features_used': dict(zip(FEATURE_NAMES, input_features))  # For debugging
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Unexpected server error: {str(e)}',
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)