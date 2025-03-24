import tensorflow as tf

def load_detection_model(model_path="models/InceptionV3_hybrid.keras"):
    model = tf.keras.models.load_model(model_path, compile=False)
    print("✅ Model loaded successfully!")
    return model
