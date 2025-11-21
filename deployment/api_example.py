"""
Example REST API for online anomaly detection inference.
Supports both Flask and FastAPI implementations.
"""
import joblib
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Optional

# Option 1: Flask implementation
try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Option 2: FastAPI implementation
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Load model and preprocessing artifacts
MODEL_PATH = 'data/model_lgbm.pkl'
SCALER_PATH = 'data/scaler.pkl'
FEATURE_NAMES_PATH = 'data/feature_names.json'

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(FEATURE_NAMES_PATH, 'r') as f:
    feature_names = json.load(f)

# Class labels
CLASS_LABELS = ['None', 'Dropoff', 'OffRoute', 'Inactive', 'Distress']


def preprocess_input(data: Dict) -> np.ndarray:
    """Preprocess input data for model inference."""
    from sklearn.preprocessing import LabelEncoder
    
    # Load encoders (in production, these should be loaded once)
    le_user = joblib.load('data/le_user.pkl')
    le_route = joblib.load('data/le_route.pkl')
    
    # Extract timestamp features
    timestamp = pd.to_datetime(data.get('timestamp', datetime.now()))
    hour = timestamp.hour
    day_of_week = timestamp.dayofweek
    month = timestamp.month
    
    # Encode categorical
    user_id_encoded = le_user.transform([str(data.get('user_id', 'unknown'))])[0]
    route_id = str(data.get('planned_route_id', 'None'))
    route_id_encoded = le_route.transform([route_id])[0]
    
    # Build feature array
    features = np.array([[
        data.get('latitude', 0.0),
        data.get('longitude', 0.0),
        data.get('speed_mps', 0.0),
        data.get('distance_from_route_m', 0.0),
        int(data.get('is_checkin', 0)),
        data.get('battery_percent', 100),
        data.get('signal_strength', 5),
        int(data.get('is_emergency_action', 0)),
        data.get('inactivity_minutes', 0),
        hour,
        day_of_week,
        month,
        user_id_encoded,
        route_id_encoded
    ]])
    
    # Scale numeric features
    numeric_cols = ['latitude', 'longitude', 'speed_mps', 'distance_from_route_m',
                    'battery_percent', 'signal_strength', 'inactivity_minutes',
                    'hour', 'day_of_week', 'month']
    numeric_indices = [feature_names.index(col) for col in numeric_cols if col in feature_names]
    features[:, numeric_indices] = scaler.transform(features[:, numeric_indices])
    
    return features


def predict_anomaly(data: Dict) -> Dict:
    """Run inference and return prediction results."""
    try:
        # Preprocess
        features = preprocess_input(data)
        
        # Predict
        probabilities = model.predict_proba(features)[0]
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        # Format response
        result = {
            'anomaly_type': CLASS_LABELS[predicted_class],
            'is_anomaly': predicted_class != 0,
            'confidence': float(confidence),
            'probabilities': {
                label: float(prob) for label, prob in zip(CLASS_LABELS, probabilities)
            }
        }
        
        return result
        
    except Exception as e:
        return {'error': str(e)}


# Flask API
if FLASK_AVAILABLE:
    app = Flask(__name__)
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'healthy'})
    
    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.json
        result = predict_anomaly(data)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
    
    @app.route('/predict/batch', methods=['POST'])
    def predict_batch():
        data_list = request.json.get('samples', [])
        results = [predict_anomaly(data) for data in data_list]
        return jsonify({'results': results})
    
    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000, debug=True)


# FastAPI implementation
if FASTAPI_AVAILABLE:
    app_fastapi = FastAPI(title="Tourist Anomaly Detection API")
    
    class AnomalyRequest(BaseModel):
        timestamp: Optional[str] = None
        user_id: str
        latitude: float
        longitude: float
        speed_mps: float = 0.0
        distance_from_route_m: float = 0.0
        planned_route_id: Optional[str] = None
        is_checkin: int = 0
        battery_percent: float = 100.0
        signal_strength: int = 5
        is_emergency_action: int = 0
        inactivity_minutes: float = 0.0
    
    class BatchRequest(BaseModel):
        samples: List[AnomalyRequest]
    
    @app_fastapi.get('/health')
    def health():
        return {'status': 'healthy'}
    
    @app_fastapi.post('/predict')
    def predict_fastapi(request: AnomalyRequest):
        data = request.dict()
        result = predict_anomaly(data)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return result
    
    @app_fastapi.post('/predict/batch')
    def predict_batch_fastapi(request: BatchRequest):
        results = [predict_anomaly(sample.dict()) for sample in request.samples]
        return {'results': results}
    
    if __name__ == '__main__':
        import uvicorn
        uvicorn.run(app_fastapi, host='0.0.0.0', port=8000)

