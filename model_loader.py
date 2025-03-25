import os
import gdown
import tensorflow as tf
from keras.models import load_model
import streamlit as st

MODEL_PATH = os.path.join(os.getcwd(), "models", "InceptionV3_hybrid.keras")
DRIVE_FILE_ID = "15QyIEgSRC38_LZooVJzDD5UZnODhyBdr"


@st.cache_resource(show_spinner="Downloading model...")
def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        try:
            gdown.download(url, MODEL_PATH, quiet=True)
            return True
        except Exception as e:
            st.error(f"Model download failed: {str(e)}")
            return False
    return True


@st.cache_resource(show_spinner="Loading detection model...")
def load_detection_model():
    if not download_model():
        st.error("Failed to download model")
        st.stop()

    try:
        return tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={
                # Add any custom layers here
                # 'CustomLayer': CustomLayer
            }
        )
    except Exception as e:
        st.error(f"""
        **Model Loading Error**:
        ```
        {str(e)}
        ```
        """)
        st.stop()