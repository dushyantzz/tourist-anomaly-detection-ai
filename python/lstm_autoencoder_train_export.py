import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib

# Data loading
X_train = np.load('./data/X_train_seq.npy')
X_test = np.load('./data/X_test_seq.npy')
y_train = np.load('./data/y_train_seq.npy')
y_test = np.load('./data/y_test_seq.npy')

# Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

joblib.dump(scaler, './deployment/scaler_seq.pkl')

# Autoencoder Model for Anomaly Detection
timesteps = X_train.shape[1]
features = X_train.shape[2]
input_dim = (timesteps, features)
inputs = Input(shape=input_dim)
encoded = LSTM(64, activation='relu', return_sequences=False)(inputs)
decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(features, activation='relu', return_sequences=True)(decoded)
autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
autoencoder.fit(X_train_scaled, X_train_scaled, epochs=50, batch_size=32, validation_split=0.2, callbacks=[es])

# Evaluation
reconstructions = autoencoder.predict(X_test_scaled)
mse = np.mean(np.power(X_test_scaled - reconstructions, 2), axis=(1,2))
threshold = np.percentile(mse, 95)
anomaly_preds = (mse > threshold).astype(int)

print('Autoencoder Classification Report:')
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, anomaly_preds))
print(confusion_matrix(y_test, anomaly_preds))

# Save model
autoencoder.save('./deployment/autoencoder_seq.h5')

# Export to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
tflite_model = converter.convert()
with open('./deployment/autoencoder_seq.tflite', 'wb') as f:
    f.write(tflite_model)
print('TFLite model exported!')
