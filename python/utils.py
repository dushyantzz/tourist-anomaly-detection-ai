"""
Utility functions for data preprocessing and model inference.
"""
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_preprocessing_artifacts(base_path='../data'):
    """Load scaler and encoders for preprocessing."""
    scaler = joblib.load(f'{base_path}/scaler.pkl')
    le_user = joblib.load(f'{base_path}/le_user.pkl')
    le_route = joblib.load(f'{base_path}/le_route.pkl')
    
    with open(f'{base_path}/feature_names.json', 'r') as f:
        feature_names = json.load(f)
    
    return scaler, le_user, le_route, feature_names

def preprocess_single_sample(data_dict, scaler, le_user, le_route, feature_names):
    """
    Preprocess a single data sample for inference.
    
    Args:
        data_dict: Dictionary with raw feature values
        scaler: Fitted StandardScaler
        le_user: Fitted LabelEncoder for user_id
        le_route: Fitted LabelEncoder for route_id
        feature_names: List of feature names in order
    
    Returns:
        Preprocessed feature array
    """
    # Extract timestamp features
    timestamp = pd.to_datetime(data_dict.get('timestamp', datetime.now()))
    hour = timestamp.hour
    day_of_week = timestamp.dayofweek
    month = timestamp.month
    
    # Encode categorical
    user_id_encoded = le_user.transform([str(data_dict.get('user_id', 'unknown'))])[0]
    route_id = str(data_dict.get('planned_route_id', 'None'))
    route_id_encoded = le_route.transform([route_id])[0]
    
    # Build feature array in correct order
    features = np.array([[
        data_dict.get('latitude', 0.0),
        data_dict.get('longitude', 0.0),
        data_dict.get('speed_mps', 0.0),
        data_dict.get('distance_from_route_m', 0.0),
        int(data_dict.get('is_checkin', 0)),
        data_dict.get('battery_percent', 100),
        data_dict.get('signal_strength', 5),
        int(data_dict.get('is_emergency_action', 0)),
        data_dict.get('inactivity_minutes', 0),
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

def create_sequences_for_lstm(data, seq_length=10):
    """
    Create sequences from time series data for LSTM.
    
    Args:
        data: Array of shape (n_samples, n_features)
        seq_length: Length of sequences
    
    Returns:
        Array of shape (n_sequences, seq_length, n_features)
    """
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

def get_anomaly_type_label(prediction, class_labels=None):
    """
    Convert prediction to anomaly type label.
    
    Args:
        prediction: Model prediction (class index or probabilities)
        class_labels: List of class labels (default: ['None', 'Dropoff', 'OffRoute', 'Inactive', 'Distress'])
    
    Returns:
        Anomaly type string
    """
    if class_labels is None:
        class_labels = ['None', 'Dropoff', 'OffRoute', 'Inactive', 'Distress']
    
    if isinstance(prediction, np.ndarray) and len(prediction.shape) > 1:
        # If probabilities, get class with highest probability
        class_idx = np.argmax(prediction)
    else:
        class_idx = int(prediction)
    
    if 0 <= class_idx < len(class_labels):
        return class_labels[class_idx]
    return 'Unknown'

