# ONNX Integration Guide for Android

This guide explains how to integrate the ONNX model for anomaly detection in your Android application using ONNX Runtime.

## Prerequisites

1. **Add ONNX Runtime dependency** to your `build.gradle` (Module: app):

```gradle
dependencies {
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.16.0'
    // Optional: GPU acceleration
    implementation 'com.microsoft.onnxruntime:onnxruntime-gpu:1.16.0'
}
```

2. **Add model to assets**: Place `model.onnx` in `app/src/main/assets/`

## Step-by-Step Integration

### 1. Create ONNX Inference Class

```kotlin
package com.tourist.anomaly.detection

import ai.onnxruntime.*
import android.content.Context
import android.util.Log
import java.nio.FloatBuffer

class ONNXAnomalyInference(context: Context, modelPath: String = "model.onnx") {
    
    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var ortSession: OrtSession? = null
    private val context: Context = context.applicationContext
    
    // Feature scaling parameters (should match Python preprocessing)
    private val featureMeans = floatArrayOf(/* ... same as TFLite ... */)
    private val featureStds = floatArrayOf(/* ... same as TFLite ... */)
    
    init {
        loadModel(modelPath)
    }
    
    private fun loadModel(modelPath: String) {
        try {
            val assetManager = context.assets
            val inputStream = assetManager.open(modelPath)
            val modelBytes = inputStream.readBytes()
            
            val sessionOptions = OrtSession.SessionOptions()
            // Optional: Enable GPU
            // sessionOptions.addCUDA()
            
            ortSession = ortEnv.createSession(modelBytes, sessionOptions)
            
            // Log model info
            val inputInfo = ortSession!!.inputNames.first()
            val outputInfo = ortSession!!.outputNames.first()
            Log.d("ONNXInference", "Model loaded - Input: $inputInfo, Output: $outputInfo")
            
        } catch (e: Exception) {
            Log.e("ONNXInference", "Error loading model: ${e.message}")
            throw e
        }
    }
    
    fun preprocessFeatures(features: Map<String, Float>): FloatArray {
        // Same preprocessing as TFLite version
        val rawFeatures = floatArrayOf(/* ... */)
        val normalizedFeatures = FloatArray(rawFeatures.size)
        for (i in rawFeatures.indices) {
            normalizedFeatures[i] = (rawFeatures[i] - featureMeans[i]) / featureStds[i]
        }
        return normalizedFeatures
    }
    
    fun predict(features: FloatArray): FloatArray {
        val session = ortSession ?: throw IllegalStateException("Model not loaded")
        
        try {
            // Get input/output names
            val inputName = session.inputNames.first()
            val outputName = session.outputNames.first()
            
            // Create input tensor: [1, n_features]
            val inputShape = longArrayOf(1, features.size.toLong())
            val inputTensor = OnnxTensor.createTensor(
                ortEnv,
                FloatBuffer.wrap(features),
                inputShape
            )
            
            // Run inference
            val inputs = mapOf(inputName to inputTensor)
            val outputs = session.run(inputs)
            
            // Get output
            val outputTensor = outputs[outputName]?.value as? Array<FloatArray>
            val result = outputTensor?.first() ?: floatArrayOf()
            
            // Cleanup
            inputTensor.close()
            outputs.close()
            
            return result
            
        } catch (e: Exception) {
            Log.e("ONNXInference", "Inference error: ${e.message}")
            throw e
        }
    }
    
    fun close() {
        ortSession?.close()
    }
}
```

### 2. Use in Your Activity

```kotlin
class MainActivity : AppCompatActivity() {
    private lateinit var onnxInference: ONNXAnomalyInference
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        try {
            onnxInference = ONNXAnomalyInference(this, "model.onnx")
        } catch (e: Exception) {
            Log.e("MainActivity", "Failed to load ONNX model: ${e.message}")
        }
    }
    
    fun detectAnomaly() {
        val features = collectFeatures()
        val preprocessed = onnxInference.preprocessFeatures(features)
        val probabilities = onnxInference.predict(preprocessed)
        
        // Get predicted class
        val predictedClass = probabilities.indices.maxByOrNull { 
            probabilities[it] 
        } ?: 0
        
        val labels = arrayOf("None", "Dropoff", "OffRoute", "Inactive", "Distress")
        val anomalyType = labels.getOrElse(predictedClass) { "Unknown" }
        
        if (predictedClass != 0) {
            handleAnomaly(anomalyType)
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        onnxInference.close()
    }
}
```

## Comparison: TFLite vs ONNX

### TFLite Advantages
- Smaller model size
- Better mobile optimization
- Lower memory footprint
- Faster inference on mobile devices

### ONNX Advantages
- Cross-platform compatibility
- Support for more model types
- Easier conversion from various frameworks
- Better for backend deployment

## Performance Tips

1. **Use NNAPI delegate** (Android Neural Networks API):
```kotlin
val sessionOptions = OrtSession.SessionOptions()
sessionOptions.addNnapi() // Enable NNAPI acceleration
```

2. **Optimize model**: Use quantization or pruning before conversion

3. **Batch processing**: Process multiple samples together

## Model Conversion Notes

The ONNX model is converted from LightGBM using `skl2onnx`. Ensure:
- Input shape matches: `[batch_size, 14]`
- Output shape: `[batch_size, 5]` (class probabilities)
- Data type: `float32`

## Troubleshooting

1. **Model loading fails**: Check model file is in assets folder
2. **Input shape error**: Verify feature preprocessing matches training
3. **Performance issues**: Enable NNAPI or GPU acceleration
4. **Memory issues**: Use model quantization

## Next Steps

- Compare TFLite vs ONNX performance on your device
- Implement model selection based on device capabilities
- Add model versioning and update mechanism

