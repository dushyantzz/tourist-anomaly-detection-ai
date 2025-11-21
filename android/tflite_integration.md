# TFLite Integration Guide for Android

This guide explains how to integrate the TFLite model for anomaly detection in your Android application.

## Prerequisites

1. **Add TensorFlow Lite dependency** to your `build.gradle` (Module: app):

```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
    // Optional: GPU acceleration
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.14.0'
}
```

2. **Add model to assets**: Place `model.tflite` in `app/src/main/assets/`

## Step-by-Step Integration

### 1. Initialize the Inference Class

```kotlin
class MainActivity : AppCompatActivity() {
    private lateinit var anomalyInference: AnomalyInference
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        try {
            anomalyInference = AnomalyInference(this, "model.tflite")
        } catch (e: Exception) {
            Log.e("MainActivity", "Failed to load model: ${e.message}")
        }
    }
}
```

### 2. Prepare Input Features

Collect sensor and location data from your app:

```kotlin
fun collectFeatures(): Map<String, Float> {
    val features = mutableMapOf<String, Float>()
    
    // Location data
    features["latitude"] = currentLocation.latitude.toFloat()
    features["longitude"] = currentLocation.longitude.toFloat()
    features["speed_mps"] = currentLocation.speed
    
    // Route data
    features["distance_from_route_m"] = calculateDistanceFromRoute()
    features["planned_route_id"] = getRouteId()
    
    // Device state
    features["battery_percent"] = getBatteryLevel().toFloat()
    features["signal_strength"] = getSignalStrength().toFloat()
    features["is_checkin"] = if (isAtCheckpoint()) 1f else 0f
    features["is_emergency_action"] = if (emergencyButtonPressed) 1f else 0f
    features["inactivity_minutes"] = getInactivityDuration()
    
    // Time features
    val calendar = Calendar.getInstance()
    features["hour"] = calendar.get(Calendar.HOUR_OF_DAY).toFloat()
    features["day_of_week"] = calendar.get(Calendar.DAY_OF_WEEK).toFloat()
    features["month"] = calendar.get(Calendar.MONTH).toFloat()
    
    // Encoded features (you'll need to implement encoding logic)
    features["user_id_encoded"] = encodeUserId(currentUserId)
    features["route_id_encoded"] = encodeRouteId(currentRouteId)
    
    return features
}
```

### 3. Run Inference

For single sample prediction (classification model):

```kotlin
fun detectAnomaly() {
    val features = collectFeatures()
    val preprocessed = anomalyInference.preprocessFeatures(features)
    val result = anomalyInference.predict(preprocessed)
    
    if (anomalyInference.isAnomaly(result)) {
        val anomalyType = anomalyInference.getAnomalyType(result)
        handleAnomaly(anomalyType)
    }
}
```

For sequence prediction (LSTM autoencoder):

```kotlin
fun detectAnomalySequence() {
    val sequenceBuffer = mutableListOf<FloatArray>()
    
    // Collect last 10 samples
    for (sample in recentSamples.takeLast(10)) {
        val preprocessed = anomalyInference.preprocessFeatures(sample)
        sequenceBuffer.add(preprocessed)
    }
    
    val reconstructionError = anomalyInference.predictSequence(
        sequenceBuffer.toTypedArray()
    )
    
    if (reconstructionError > ANOMALY_THRESHOLD) {
        handleAnomaly("Anomaly detected")
    }
}
```

### 4. Handle Anomaly Detection Results

```kotlin
fun handleAnomaly(anomalyType: String) {
    when (anomalyType) {
        "Distress" -> {
            // Trigger emergency alert
            sendEmergencyAlert()
        }
        "Dropoff" -> {
            // User may have left the route
            notifyUser("You seem to have left your planned route")
        }
        "Inactive" -> {
            // User inactive for too long
            checkUserStatus()
        }
        "OffRoute" -> {
            // User is off the planned route
            suggestRouteCorrection()
        }
    }
}
```

## Preprocessing Requirements

The model expects standardized features. The `AnomalyInference` class handles this automatically, but you need to ensure:

1. **Feature scaling**: Mean and std values should match training data
2. **Categorical encoding**: User IDs and route IDs must be encoded consistently
3. **Time features**: Extract hour, day_of_week, month from timestamp

## Model Input/Output

### Classification Model (LightGBM → ONNX/TFLite)
- **Input**: `[1, 14]` - Single sample with 14 features
- **Output**: `[1, 5]` - Class probabilities for 5 classes

### Autoencoder Model (LSTM)
- **Input**: `[1, 10, 13]` - Sequence of 10 timesteps, 13 features each
- **Output**: `[1, 10, 13]` - Reconstructed sequence
- **Anomaly detection**: Calculate MSE between input and output

## Performance Optimization

1. **Use GPU delegate** (if available):
```kotlin
val options = Interpreter.Options()
options.addDelegate(GpuDelegate())
interpreter = Interpreter(modelBuffer, options)
```

2. **Batch processing**: Process multiple samples together when possible

3. **Async inference**: Run inference on background thread:
```kotlin
lifecycleScope.launch(Dispatchers.Default) {
    val result = anomalyInference.predict(features)
    withContext(Dispatchers.Main) {
        handleResult(result)
    }
}
```

## Testing

Test with sample data:

```kotlin
@Test
fun testAnomalyDetection() {
    val features = mapOf(
        "latitude" to 26.8f,
        "longitude" to 92.9f,
        "speed_mps" to 0f,
        "distance_from_route_m" to 100f,
        "battery_percent" to 10f,
        // ... other features
    )
    
    val result = anomalyInference.predict(
        anomalyInference.preprocessFeatures(features)
    )
    
    assertTrue(anomalyInference.isAnomaly(result))
}
```

## Troubleshooting

1. **Model not found**: Ensure `model.tflite` is in `assets/` folder
2. **Input shape mismatch**: Check feature order matches training
3. **Low accuracy**: Verify preprocessing matches Python training pipeline
4. **Performance issues**: Use GPU delegate or optimize model quantization

## Next Steps

- Implement user/route ID encoding logic
- Add real-time feature collection from sensors
- Integrate with location services
- Add model update mechanism for retraining

