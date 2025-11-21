"""
Train LSTM Autoencoder model for tourist anomaly detection.
This script replicates the training workflow from the notebook.
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import json
import os

def create_sequences(data, seq_length=10):
    """Create sequences from time series data for LSTM."""
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

def train_lstm_autoencoder():
    """Train LSTM autoencoder for anomaly detection."""
    
    print("=" * 60)
    print("Training LSTM Autoencoder Model")
    print("=" * 60)
    
    # Load and prepare data
    print("\n1. Loading data...")
    df = pd.read_csv('../data/tourism_anomaly_dataset.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"   Dataset shape: {df.shape}")
    
    # Feature engineering
    print("\n2. Feature engineering...")
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Encode categorical
    le_user = LabelEncoder()
    le_route = LabelEncoder()
    df['user_id_encoded'] = le_user.fit_transform(df['user_id'].astype(str))
    df['planned_route_id'] = df['planned_route_id'].astype(str).fillna('None')
    df['route_id_encoded'] = le_route.fit_transform(df['planned_route_id'])
    
    # Select features
    feature_cols = [
        'latitude', 'longitude', 'speed_mps', 'distance_from_route_m',
        'is_checkin', 'battery_percent', 'signal_strength',
        'is_emergency_action', 'inactivity_minutes',
        'hour', 'day_of_week', 'user_id_encoded', 'route_id_encoded'
    ]
    
    X = df[feature_cols].copy()
    y = (df['anomaly_type'].notna()).astype(int)
    
    print(f"   Anomaly rate: {y.mean():.2%}")
    
    # Scale features
    print("\n3. Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler
    os.makedirs('../data', exist_ok=True)
    joblib.dump(scaler, '../data/scaler_lstm.pkl')
    with open('../data/feature_names_lstm.json', 'w') as f:
        json.dump(feature_cols, f)
    print("   ✅ Scaler and feature names saved")
    
    # Create sequences
    print("\n4. Creating sequences...")
    seq_length = 10
    X_seq = create_sequences(X_scaled, seq_length)
    
    # Split data (use normal data for training)
    normal_mask = y == 0
    X_normal = X_scaled[normal_mask]
    X_anomaly = X_scaled[~normal_mask]
    
    X_normal_seq = create_sequences(X_normal, seq_length)
    X_anomaly_seq = create_sequences(X_anomaly, seq_length)
    
    X_train_seq, X_val_seq = train_test_split(
        X_normal_seq, test_size=0.2, random_state=42
    )
    
    print(f"   Training sequences (normal): {X_train_seq.shape}")
    print(f"   Validation sequences (normal): {X_val_seq.shape}")
    print(f"   Anomaly sequences (test): {X_anomaly_seq.shape}")
    
    # Define model
    print("\n5. Building LSTM Autoencoder...")
    input_dim = X_train_seq.shape[2]
    latent_dim = 32
    
    # Encoder
    encoder_input = layers.Input(shape=(seq_length, input_dim))
    encoder_lstm1 = layers.LSTM(64, return_sequences=True)(encoder_input)
    encoder_lstm2 = layers.LSTM(32, return_sequences=False)(encoder_lstm1)
    encoder_output = layers.Dense(latent_dim, activation='relu')(encoder_lstm2)
    
    # Decoder
    decoder_repeat = layers.RepeatVector(seq_length)(encoder_output)
    decoder_lstm1 = layers.LSTM(32, return_sequences=True)(decoder_repeat)
    decoder_lstm2 = layers.LSTM(64, return_sequences=True)(decoder_lstm1)
    decoder_output = layers.TimeDistributed(layers.Dense(input_dim))(decoder_lstm2)
    
    # Autoencoder
    autoencoder = models.Model(encoder_input, decoder_output)
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print(f"   Model input shape: {autoencoder.input_shape}")
    print(f"   Model output shape: {autoencoder.output_shape}")
    
    # Train model
    print("\n6. Training model...")
    print("   This may take several minutes...")
    
    history = autoencoder.fit(
        X_train_seq, X_train_seq,
        epochs=30,  # Reduced for faster training
        batch_size=32,
        validation_data=(X_val_seq, X_val_seq),
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
        ]
    )
    
    print("   ✅ Model trained")
    
    # Evaluate
    print("\n7. Evaluating model...")
    val_pred = autoencoder.predict(X_val_seq, verbose=0)
    anomaly_pred = autoencoder.predict(X_anomaly_seq, verbose=0)
    
    val_errors = np.mean(np.square(X_val_seq - val_pred), axis=(1, 2))
    anomaly_errors = np.mean(np.square(X_anomaly_seq - anomaly_pred), axis=(1, 2))
    
    threshold = np.percentile(val_errors, 95)
    
    print(f"   Reconstruction error threshold: {threshold:.4f}")
    print(f"   Normal data errors - Mean: {val_errors.mean():.4f}, Max: {val_errors.max():.4f}")
    print(f"   Anomaly data errors - Mean: {anomaly_errors.mean():.4f}, Max: {anomaly_errors.max():.4f}")
    
    # Save model
    print("\n8. Saving model...")
    autoencoder.save('../data/model_lstm_autoencoder.h5')
    np.save('../data/anomaly_threshold.npy', threshold)
    
    print("   ✅ Model saved as model_lstm_autoencoder.h5")
    print("   ✅ Threshold saved as anomaly_threshold.npy")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    return autoencoder, threshold

if __name__ == '__main__':
    train_lstm_autoencoder()

