# Mobile Deployment Guide

This guide covers deploying the anomaly detection models to mobile applications (Android/iOS) and syncing with backend services.

## Model Deployment Options

### 1. On-Device Inference (TFLite/ONNX)

**Advantages:**
- Works offline
- Low latency
- Privacy-preserving (data stays on device)
- No network dependency

**Use Cases:**
- Real-time anomaly detection
- Emergency situations with poor connectivity
- Privacy-sensitive applications

**Implementation:**
- See `android/tflite_integration.md` for TFLite setup
- See `android/onnx_integration.md` for ONNX setup

### 2. Backend API Inference

**Advantages:**
- Easy model updates
- Centralized logging and analytics
- Can use more complex models
- Consistent preprocessing

**Use Cases:**
- When model updates are frequent
- When you need centralized monitoring
- When device resources are limited

**Implementation:**
- See `deployment/api_example.py` for Flask/FastAPI setup

### 3. Hybrid Approach (Recommended)

Use on-device for real-time detection, sync with backend for:
- Model updates
- Logging and analytics
- Emergency alerts
- Model retraining data collection

## Model Update Strategy

### Option A: Over-the-Air (OTA) Updates

```kotlin
// Android example
class ModelUpdateManager(private val context: Context) {
    private val modelVersionPrefs = context.getSharedPreferences("model_prefs", Context.MODE_PRIVATE)
    
    suspend fun checkForUpdates(): Boolean {
        val currentVersion = modelVersionPrefs.getInt("model_version", 0)
        val latestVersion = apiService.getLatestModelVersion()
        
        if (latestVersion > currentVersion) {
            downloadModel(latestVersion)
            return true
        }
        return false
    }
    
    private suspend fun downloadModel(version: Int) {
        val modelBytes = apiService.downloadModel(version)
        saveModelToAssets(modelBytes, "model_v$version.tflite")
        modelVersionPrefs.edit().putInt("model_version", version).apply()
    }
}
```

### Option B: App Update

Include model in app bundle and update through Play Store/App Store.

## Data Collection for Retraining

### Collecting Inference Data

```kotlin
data class InferenceLog(
    val timestamp: Long,
    val features: Map<String, Float>,
    val prediction: String,
    val confidence: Float,
    val actualLabel: String? = null  // If user provides feedback
)

class InferenceLogger {
    private val logs = mutableListOf<InferenceLog>()
    
    fun logInference(features: Map<String, Float>, result: AnomalyResult) {
        logs.add(InferenceLog(
            timestamp = System.currentTimeMillis(),
            features = features,
            prediction = result.anomalyType,
            confidence = result.confidence
        ))
        
        // Sync to backend periodically
        if (logs.size >= 100) {
            syncToBackend()
        }
    }
    
    private suspend fun syncToBackend() {
        apiService.uploadInferenceLogs(logs)
        logs.clear()
    }
}
```

## Backend Integration

### 1. Model Serving

Deploy the API (see `deployment/api_example.py`):

```bash
# Using Flask
python deployment/api_example.py

# Using FastAPI
pip install fastapi uvicorn
uvicorn deployment.api_example:app_fastapi --host 0.0.0.0 --port 8000
```

### 2. Model Versioning

```python
# models/
#   v1/
#     model_lgbm.pkl
#     scaler.pkl
#   v2/
#     model_lgbm.pkl
#     scaler.pkl

class ModelManager:
    def __init__(self):
        self.current_version = "v2"
        self.models = {}
        self.load_model(self.current_version)
    
    def load_model(self, version: str):
        model_path = f"models/{version}/model_lgbm.pkl"
        self.models[version] = joblib.load(model_path)
```

### 3. A/B Testing

Test new models before full deployment:

```python
@app.route('/predict', methods=['POST'])
def predict():
    user_id = request.json.get('user_id')
    
    # Route 10% of users to new model
    if hash(user_id) % 10 == 0:
        model = model_manager.get_model('v2')
    else:
        model = model_manager.get_model('v1')
    
    # ... inference ...
```

## Monitoring and Analytics

### Key Metrics to Track

1. **Inference Performance**
   - Latency (on-device vs API)
   - Model accuracy (if ground truth available)
   - False positive/negative rates

2. **Model Usage**
   - Number of predictions per day
   - Geographic distribution
   - Anomaly detection rates

3. **System Health**
   - Model load time
   - Memory usage
   - Battery impact

### Example Monitoring Dashboard

```python
# Backend monitoring endpoint
@app.route('/metrics', methods=['GET'])
def get_metrics():
    return jsonify({
        'total_predictions': get_total_predictions(),
        'anomaly_rate': get_anomaly_rate(),
        'avg_latency_ms': get_avg_latency(),
        'model_version': current_model_version,
        'active_users': get_active_users()
    })
```

## Security Considerations

### 1. Model Protection

- Obfuscate model files
- Use encrypted model storage
- Validate model integrity before loading

### 2. API Security

- Use authentication tokens
- Rate limiting
- Input validation
- HTTPS only

### 3. Data Privacy

- Minimize data collection
- Anonymize user data
- Comply with GDPR/local regulations
- Allow users to opt-out

## Performance Optimization

### 1. Model Quantization

Reduce model size and improve inference speed:

```python
# Quantize TFLite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]  # or tf.int8
tflite_model = converter.convert()
```

### 2. Caching

Cache frequent predictions:

```kotlin
class PredictionCache {
    private val cache = LRUCache<String, AnomalyResult>(100)
    
    fun getCached(features: Map<String, Float>): AnomalyResult? {
        val key = features.hashCode().toString()
        return cache.get(key)
    }
    
    fun cache(features: Map<String, Float>, result: AnomalyResult) {
        val key = features.hashCode().toString()
        cache.put(key, result)
    }
}
```

### 3. Batch Processing

Process multiple samples together when possible.

## Testing Strategy

### 1. Unit Tests

```kotlin
@Test
fun testAnomalyDetection() {
    val features = createTestFeatures()
    val result = anomalyInference.predict(features)
    assertTrue(anomalyInference.isAnomaly(result))
}
```

### 2. Integration Tests

Test end-to-end flow:
- Feature collection → Preprocessing → Inference → Result handling

### 3. Performance Tests

Measure:
- Inference latency
- Memory usage
- Battery consumption

## Deployment Checklist

- [ ] Model files included in app/assets
- [ ] Preprocessing logic matches training
- [ ] Error handling implemented
- [ ] Logging and analytics set up
- [ ] Model update mechanism tested
- [ ] Security measures in place
- [ ] Performance optimized
- [ ] User privacy considered
- [ ] Documentation updated
- [ ] Testing completed

## Troubleshooting

### Common Issues

1. **Model not found**: Check assets folder and file paths
2. **Input shape mismatch**: Verify preprocessing matches training
3. **Low accuracy**: Check feature scaling and encoding
4. **Performance issues**: Enable GPU/NNAPI, use quantization
5. **Memory issues**: Reduce batch size, use model quantization

## Next Steps

1. Set up CI/CD for model updates
2. Implement A/B testing framework
3. Add real-time monitoring dashboard
4. Create model retraining pipeline
5. Implement user feedback mechanism

