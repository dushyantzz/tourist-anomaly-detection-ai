# Tourist Anomaly Detection AI

## Overview

This repository enables AI-driven anomaly detection for tourism/emergency applications. It provides:

- End-to-end workflows for classic ML and deep learning (LSTM/autoencoder) anomaly detection
- Data processing, feature engineering, and ready-to-use training notebooks/scripts
- Model export to mobile (TFLite/ONNX) and backend
- Android (Kotlin) integration samples
- Example datasets and deployment guides

## Folder Structure

```
tourist-anomaly-detection-ai/
├── data/                  # Example datasets, preprocessing configs
├── notebooks/             # Jupyter/Colab notebooks for all experiments
│   ├── train_classic_ml.ipynb
│   └── train_lstm_autoencoder.ipynb
├── python/                # Python scripts for CLI/automation
│   ├── preprocess_data.py
│   ├── train_model.py
│   ├── export_onnx.py
│   ├── export_tflite.py
│   └── utils.py
├── android/               # Mobile integration
│   ├── AnomalyInference.kt
│   ├── onnx_integration.md
│   └── tflite_integration.md
├── deployment/            # Backend/API, model deployment tips
│   ├── api_example.py
│   └── mobile_deployment_guide.md
└── README.md
```

## Step-by-Step Usage

### 1. Data Preparation

- Place your CSV dataset in `data/` (e.g., `data/tourism_anomaly_dataset.csv`)
- **Required columns**: `timestamp`, `user_id`, `latitude`, `longitude`, `speed_mps`, `planned_route_id`, `distance_from_route_m`, `is_checkin`, `battery_percent`, `signal_strength`, `is_emergency_action`, `inactivity_minutes`, `anomaly_type`
- **Optional**:
  - Run `python/python/preprocess_data.py` to clean, encode, and scale your data.

### 2. Classic ML (Tabular) Anomaly Detection

Use `notebooks/train_classic_ml.ipynb` (Colab or Jupyter) for:

- LightGBM/XGBoost modeling
- Stratified train/test split
- Feature engineering (timing, route, scaling)
- Model evaluation (classification report, confusion matrix)
- Model export (`model_lgbm.pkl`, `model.onnx`)
- All steps ready-to-run and notebook-explained.

### 3. Deep Learning Workflow (LSTM/Autoencoder)

Use `notebooks/train_lstm_autoencoder.ipynb` for:

- Sequence modeling for time-based anomalies (LSTM/GRU)
- Unsupervised anomaly detection (autoencoder learns "normal", anomalies = high error)
- TensorFlow/Keras model training and TFLite export
- Model evaluation and visualization

### 4. Model Export (For App Integration)

**For TFLite:**
- Run/export using code in `python/export_tflite.py` or via notebook.
- Output: `model.tflite` in repo

**For ONNX (tabular ML):**
- Use `python/export_onnx.py`
- Output: `model.onnx`

### 5. Mobile App (Android/Kotlin) Integration

Go to `/android`

- `AnomalyInference.kt`: Kotlin class for loading and running TFLite/ONNX models
- `onnx_integration.md` and `tflite_integration.md`: Step-by-step Android/ML libraries, preprocessing, and inference pipeline setup
- Example: Replicate scaler/encoder logic in Kotlin if used in Python

### 6. Backend/API Deployment

- `/deployment/api_example.py`: Example REST API for online inference (Flask/FastAPI)
- `/deployment/mobile_deployment_guide.md`: How to serve updates and sync with app
- Details for both on-device and backend serving covered

### 7. Update and Retrain

- Place new logs/data in `data/`
- Rerun notebooks/scripts
- Propose improvements via issues or pull requests

## Model Pipeline (Quick Glance)

```
Prepare dataset →
Feature process/encode/scale →
Train (LightGBM/XGBoost or LSTM/autoencoder) →
Evaluate and tune →
Export (.onnx/.tflite) →
Integrate in Android or backend →
Detect anomalies in real time on device!
```

## Tips for SIH/Dev Submissions

- All code has comments and modular structure
- Each model export is tested and can be rapidly replaced for new data/emergencies
- Android and backend guides help non-ML devs plug-and-play the solution
- Use provided notebooks as templates for further experiments

## Issues, Help, and Contributions

- Use GitHub Issues for bugs or adaptation needs
- Fork, commit, and submit PRs for downstream use (e.g., Bluetooth mesh, UI enhancements)
- For production: regularly retrain on real usage data

## License

MIT (or adapt for your hackathon/enterprise as needed)

---

This repo is tailored for rapid SIH/production deployment and can power both demos and real deployments!
