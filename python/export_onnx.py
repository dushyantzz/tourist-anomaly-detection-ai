"""
Export LightGBM model to ONNX format for mobile/backend deployment.
"""
import joblib
import numpy as np
import json
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from onnxruntime import InferenceSession
import onnx

def export_lgbm_to_onnx(model_path='data/model_lgbm.pkl', 
                        output_path='data/model.onnx',
                        feature_names_path='data/feature_names.json'):
    """
    Convert LightGBM model to ONNX format.
    
    Args:
        model_path: Path to saved LightGBM model (.pkl)
        output_path: Output path for ONNX model
        feature_names_path: Path to feature names JSON file
    """
    print("Loading LightGBM model...")
    model = joblib.load(model_path)
    
    # Load feature names
    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)
    
    print(f"Model has {len(feature_names)} features")
    print(f"Features: {feature_names}")
    
    # Create input type specification
    n_features = len(feature_names)
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    
    print("Converting to ONNX...")
    try:
        onnx_model = to_onnx(model, initial_types=initial_type, target_opset=12)
        
        # Save ONNX model
        with open(output_path, 'wb') as f:
            f.write(onnx_model.SerializeToString())
    except Exception as e:
        print(f"Error converting to ONNX: {e}")
        print("Note: ONNX conversion may require additional dependencies or model format adjustments.")
        print("You can use the TFLite model for mobile deployment instead.")
        raise
    print(f"✅ ONNX model saved to {output_path}")
    
    # Verify the model
    print("\nVerifying ONNX model...")
    session = InferenceSession(output_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Test with dummy input
    dummy_input = np.random.randn(1, n_features).astype(np.float32)
    result = session.run([output_name], {input_name: dummy_input})
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {result[0].shape}")
    print(f"✅ ONNX model verified successfully!")
    
    return output_path

if __name__ == '__main__':
    export_lgbm_to_onnx()
