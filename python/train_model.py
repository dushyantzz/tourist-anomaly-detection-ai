"""
Train LightGBM model for tourist anomaly detection.
This script replicates the training workflow from the notebook.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from lightgbm import LGBMClassifier
import joblib
import json
import os

def train_classic_ml_model():
    """Train LightGBM classifier for anomaly detection."""
    
    print("=" * 60)
    print("Training Classic ML Model (LightGBM)")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv('../data/tourism_anomaly_dataset.csv')
    print(f"   Dataset shape: {df.shape}")
    print(f"   Anomaly types: {df['anomaly_type'].unique()}")
    
    # Feature engineering
    print("\n2. Feature engineering...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    # Encode categorical variables
    le_user = LabelEncoder()
    le_route = LabelEncoder()
    df['user_id_encoded'] = le_user.fit_transform(df['user_id'].astype(str))
    df['planned_route_id'] = df['planned_route_id'].astype(str).fillna('None')
    df['route_id_encoded'] = le_route.fit_transform(df['planned_route_id'])
    
    # Encode boolean flags
    for col in ['is_checkin', 'is_emergency_action']:
        df[col] = df[col].astype(int)
    
    # Prepare target variable
    df['anomaly_type_encoded'] = LabelEncoder().fit_transform(df['anomaly_type'].fillna('None'))
    print(f"   Anomaly rate: {(df['anomaly_type'].notna()).mean():.2%}")
    
    # Select features
    print("\n3. Preparing features...")
    feature_cols = [
        'latitude', 'longitude', 'speed_mps', 'distance_from_route_m',
        'is_checkin', 'battery_percent', 'signal_strength', 
        'is_emergency_action', 'inactivity_minutes',
        'hour', 'day_of_week', 'month',
        'user_id_encoded', 'route_id_encoded'
    ]
    
    X = df[feature_cols].copy()
    y = df['anomaly_type_encoded'].copy()
    
    # Scale numeric columns
    numeric_cols = ['latitude', 'longitude', 'speed_mps', 'distance_from_route_m',
                    'battery_percent', 'signal_strength', 'inactivity_minutes',
                    'hour', 'day_of_week', 'month']
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    # Train/test split
    print("\n4. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    
    # Save preprocessing artifacts
    print("\n5. Saving preprocessing artifacts...")
    os.makedirs('../data', exist_ok=True)
    joblib.dump(scaler, '../data/scaler.pkl')
    joblib.dump(le_user, '../data/le_user.pkl')
    joblib.dump(le_route, '../data/le_route.pkl')
    with open('../data/feature_names.json', 'w') as f:
        json.dump(feature_cols, f)
    print("   ✅ Preprocessing artifacts saved")
    
    # Train model
    print("\n6. Training LightGBM model...")
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    print("   ✅ Model trained")
    
    # Evaluate
    print("\n7. Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n   Accuracy: {accuracy:.4f}")
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\n   Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    print("\n8. Saving model and feature importance...")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    feature_importance.to_csv('../data/feature_importances.csv', index=False)
    joblib.dump(model, '../data/model_lgbm.pkl')
    
    print("\n   Top 5 features:")
    print(feature_importance.head().to_string(index=False))
    print("\n   ✅ Model saved as model_lgbm.pkl")
    print("   ✅ Feature importance saved as feature_importances.csv")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    return model, scaler, le_user, le_route

if __name__ == '__main__':
    train_classic_ml_model()
