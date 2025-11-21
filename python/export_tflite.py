"""
Export LSTM Autoencoder model to TFLite format for Android deployment.
"""
import tensorflow as tf
import numpy as np
import json

def export_lstm_to_tflite(model_path='data/model_lstm_autoencoder.h5',
                          output_path='data/model.tflite',
                          feature_names_path='data/feature_names_lstm.json'):
    """
    Convert Keras LSTM model to TFLite format.
    
    Args:
        model_path: Path to saved Keras model (.h5)
        output_path: Output path for TFLite model
        feature_names_path: Path to feature names JSON file
    """
    print("Loading Keras model...")
    model = tf.keras.models.load_model(model_path)
    
    # Load feature names
    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)
    
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    print(f"Features: {feature_names}")
    
    # Convert to TFLite
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimize for size and latency
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save TFLite model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✅ TFLite model saved to {output_path}")
    
    # Get model size
    size_kb = len(tflite_model) / 1024
    print(f"Model size: {size_kb:.2f} KB")
    
    # Verify the model
    print("\nVerifying TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input details: {input_details[0]}")
    print(f"Output details: {output_details[0]}")
    
    # Test with dummy input
    seq_length = model.input_shape[1]
    n_features = model.input_shape[2]
    dummy_input = np.random.randn(1, seq_length, n_features).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"Test input shape: {dummy_input.shape}")
    print(f"Test output shape: {output.shape}")
    print(f"✅ TFLite model verified successfully!")
    
    return output_path

if __name__ == '__main__':
    export_lstm_to_tflite()
