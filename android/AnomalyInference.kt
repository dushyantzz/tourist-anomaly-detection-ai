package com.tourist.anomaly.detection

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.sqrt

/**
 * Kotlin class for running anomaly detection inference using TFLite model.
 * 
 * This class handles:
 * - Loading TFLite model from assets
 * - Preprocessing input features
 * - Running inference
 * - Post-processing results (reconstruction error for autoencoder)
 */
class AnomalyInference(context: Context, modelPath: String = "model.tflite") {
    
    private var interpreter: Interpreter? = null
    private val context: Context = context.applicationContext
    
    // Feature scaling parameters (should match Python preprocessing)
    // These should be loaded from a config file or passed as parameters
    private val featureMeans = floatArrayOf(
        26.8f,  // latitude mean
        92.9f,  // longitude mean
        0.9f,   // speed_mps mean
        25.0f,  // distance_from_route_m mean
        0.5f,   // is_checkin mean
        60.0f,  // battery_percent mean
        3.0f,   // signal_strength mean
        0.1f,   // is_emergency_action mean
        10.0f,  // inactivity_minutes mean
        12.0f,  // hour mean
        3.0f,   // day_of_week mean
        1.0f,   // month mean
        25.0f,  // user_id_encoded mean
        1.0f    // route_id_encoded mean
    )
    
    private val featureStds = floatArrayOf(
        0.4f,   // latitude std
        0.5f,   // longitude std
        0.8f,   // speed_mps std
        30.0f,  // distance_from_route_m std
        0.5f,   // is_checkin std
        25.0f,  // battery_percent std
        1.5f,   // signal_strength std
        0.3f,   // is_emergency_action std
        15.0f,  // inactivity_minutes std
        6.0f,   // hour std
        2.0f,   // day_of_week std
        0.0f,   // month std
        15.0f,  // user_id_encoded std
        1.0f    // route_id_encoded std
    )
    
    // Anomaly threshold (should be loaded from model training)
    private val anomalyThreshold = 0.5f
    
    init {
        loadModel(modelPath)
    }
    
    /**
     * Load TFLite model from assets folder.
     */
    private fun loadModel(modelPath: String) {
        try {
            val assetManager = context.assets
            val inputStream = assetManager.open(modelPath)
            val modelBuffer = inputStream.readBytes()
            
            interpreter = Interpreter(modelBuffer)
            
            // Log model input/output details
            val inputTensor = interpreter!!.getInputTensor(0)
            val outputTensor = interpreter!!.getOutputTensor(0)
            
            android.util.Log.d("AnomalyInference", 
                "Model loaded - Input: ${inputTensor.shape().contentToString()}, " +
                "Output: ${outputTensor.shape().contentToString()}")
        } catch (e: Exception) {
            android.util.Log.e("AnomalyInference", "Error loading model: ${e.message}")
            throw e
        }
    }
    
    /**
     * Preprocess input features to match model requirements.
     * 
     * @param features Map of feature names to values
     * @return Preprocessed feature array ready for model input
     */
    fun preprocessFeatures(features: Map<String, Float>): FloatArray {
        // Extract features in correct order
        val rawFeatures = floatArrayOf(
            features["latitude"] ?: 0f,
            features["longitude"] ?: 0f,
            features["speed_mps"] ?: 0f,
            features["distance_from_route_m"] ?: 0f,
            if (features["is_checkin"] == 1f) 1f else 0f,
            features["battery_percent"] ?: 100f,
            features["signal_strength"] ?: 5f,
            if (features["is_emergency_action"] == 1f) 1f else 0f,
            features["inactivity_minutes"] ?: 0f,
            features["hour"] ?: 12f,
            features["day_of_week"] ?: 0f,
            features["month"] ?: 1f,
            features["user_id_encoded"] ?: 0f,
            features["route_id_encoded"] ?: 0f
        )
        
        // Standardize features: (x - mean) / std
        val normalizedFeatures = FloatArray(rawFeatures.size)
        for (i in rawFeatures.indices) {
            normalizedFeatures[i] = (rawFeatures[i] - featureMeans[i]) / featureStds[i]
        }
        
        return normalizedFeatures
    }
    
    /**
     * Run inference on a single sample (for classification model).
     * 
     * @param features Preprocessed feature array
     * @return Prediction result with class probabilities
     */
    fun predict(features: FloatArray): AnomalyResult {
        val interpreter = this.interpreter ?: throw IllegalStateException("Model not loaded")
        
        // Prepare input buffer
        val inputShape = interpreter.getInputTensor(0).shape()
        val inputBuffer = ByteBuffer.allocateDirect(4 * inputShape[0] * inputShape[1])
        inputBuffer.order(ByteOrder.nativeOrder())
        
        // For single sample, reshape if needed
        if (inputShape.size == 2) {
            // Tabular model: [1, n_features]
            for (value in features) {
                inputBuffer.putFloat(value)
            }
        } else if (inputShape.size == 3) {
            // Sequence model: [1, seq_length, n_features]
            // For single prediction, repeat the features
            val seqLength = inputShape[1]
            for (i in 0 until seqLength) {
                for (value in features) {
                    inputBuffer.putFloat(value)
                }
            }
        }
        inputBuffer.rewind()
        
        // Prepare output buffer
        val outputShape = interpreter.getOutputTensor(0).shape()
        val outputBuffer = ByteBuffer.allocateDirect(4 * outputShape.reduce { a, b -> a * b })
        outputBuffer.order(ByteOrder.nativeOrder())
        
        // Run inference
        interpreter.run(inputBuffer, outputBuffer)
        outputBuffer.rewind()
        
        // Read output
        val outputSize = outputShape.reduce { a, b -> a * b }
        val output = FloatArray(outputSize)
        for (i in 0 until outputSize) {
            output[i] = outputBuffer.float
        }
        
        return AnomalyResult(output, features)
    }
    
    /**
     * Run inference on sequence data (for LSTM autoencoder).
     * 
     * @param sequences Array of feature arrays (sequence_length x n_features)
     * @return Reconstruction error (higher = more anomalous)
     */
    fun predictSequence(sequences: Array<FloatArray>): Float {
        val interpreter = this.interpreter ?: throw IllegalStateException("Model not loaded")
        
        val inputShape = interpreter.getInputTensor(0).shape()
        val seqLength = inputShape[1]
        val nFeatures = inputShape[2]
        
        // Prepare input buffer: [1, seq_length, n_features]
        val inputBuffer = ByteBuffer.allocateDirect(4 * seqLength * nFeatures)
        inputBuffer.order(ByteOrder.nativeOrder())
        
        for (seq in sequences.take(seqLength)) {
            for (value in seq) {
                inputBuffer.putFloat(value)
            }
        }
        inputBuffer.rewind()
        
        // Prepare output buffer
        val outputShape = interpreter.getOutputTensor(0).shape()
        val outputBuffer = ByteBuffer.allocateDirect(4 * outputShape.reduce { a, b -> a * b })
        outputBuffer.order(ByteOrder.nativeOrder())
        
        // Run inference
        interpreter.run(inputBuffer, outputBuffer)
        outputBuffer.rewind()
        
        // Calculate reconstruction error (MSE)
        var mse = 0f
        val outputSize = outputShape.reduce { a, b -> a * b }
        for (i in 0 until outputSize) {
            val predicted = outputBuffer.float
            val actual = inputBuffer.getFloat(i * 4) // Get original input
            val diff = predicted - actual
            mse += diff * diff
        }
        mse /= outputSize
        
        return mse
    }
    
    /**
     * Check if a prediction indicates an anomaly.
     */
    fun isAnomaly(result: AnomalyResult): Boolean {
        // For classification: check if predicted class is not "None" (class 0)
        // For autoencoder: check if reconstruction error exceeds threshold
        if (result.probabilities.size > 1) {
            // Classification model
            val predictedClass = result.probabilities.indices.maxByOrNull { 
                result.probabilities[it] 
            } ?: 0
            return predictedClass != 0 // 0 = None (normal)
        } else {
            // Autoencoder: use reconstruction error
            return result.reconstructionError > anomalyThreshold
        }
    }
    
    /**
     * Get anomaly type label from prediction.
     */
    fun getAnomalyType(result: AnomalyResult): String {
        val labels = arrayOf("None", "Dropoff", "OffRoute", "Inactive", "Distress")
        
        if (result.probabilities.size > 1) {
            val predictedClass = result.probabilities.indices.maxByOrNull { 
                result.probabilities[it] 
            } ?: 0
            return labels.getOrElse(predictedClass) { "Unknown" }
        }
        
        return if (isAnomaly(result)) "Anomaly" else "None"
    }
    
    /**
     * Clean up resources.
     */
    fun close() {
        interpreter?.close()
        interpreter = null
    }
}

/**
 * Data class to hold inference results.
 */
data class AnomalyResult(
    val probabilities: FloatArray,
    val inputFeatures: FloatArray,
    val reconstructionError: Float = 0f
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        
        other as AnomalyResult
        
        if (!probabilities.contentEquals(other.probabilities)) return false
        if (!inputFeatures.contentEquals(other.inputFeatures)) return false
        if (reconstructionError != other.reconstructionError) return false
        
        return true
    }
    
    override fun hashCode(): Int {
        var result = probabilities.contentHashCode()
        result = 31 * result + inputFeatures.contentHashCode()
        result = 31 * result + reconstructionError.hashCode()
        return result
    }
}

