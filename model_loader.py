import os
import gdown
from tensorflow.keras.models import load_model
from keras.src.saving import saving_lib
from tensorflow.keras import Model

MODEL_PATH = "models/InceptionV3_hybrid.h5"
DRIVE_FILE_ID = "1XTXWWF8wxFGRAH30VVRIDVfwniivdQ9t"

def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        print("Downloading model from Google Drive...")
        gdown.download(url, MODEL_PATH, quiet=False)
    else:
        print("Model already downloaded.")

def load_detection_model():
    download_model()
    try:
        return load_model(MODEL_PATH, compile=False)
    except TypeError as e:
        print("❌ TypeError while loading the model:")
        print(e)
        raise e
