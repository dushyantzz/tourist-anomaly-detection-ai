def export_tflite(keras_model_path, tflite_out_path):
    import tensorflow as tf
    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_out_path, 'wb') as f:
        f.write(tflite_model)
    print(f'TFLite model exported to {tflite_out_path}')

if __name__ == "__main__":
    export_tflite('./deployment/autoencoder_seq.h5', './deployment/autoencoder_seq.tflite')
