import streamlit as st
import os
import numpy as np
from tempfile import NamedTemporaryFile

# Set page config first
st.set_page_config(
    page_title="Deepfake Detector",
    layout="centered",
    menu_items={
        'Get Help': 'https://github.com/your-repo/issues',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# Deepfake Detection App"
    }
)

# Initialize session state
if 'model' not in st.session_state:
    from model_loader import load_detection_model
    st.session_state.model = load_detection_model()

# Import other components after model is loaded
from config import UPLOAD_FOLDER, FAKE_THRESHOLD, REAL_THRESHOLD, NUM_FRAMES
from utils.file_utils import allowed_file
from utils.frame_extractor import extract_frames
from utils.image_utils import prepare_image

# Create upload folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# App UI
st.title("🧠 Deepfake Detection App")
st.write("Upload an image or video to detect if it's real or fake.")

# File uploader
uploaded_file = st.file_uploader(
    "📂 Choose an image or video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
    help="Maximum file size: 200MB"
)

if uploaded_file is not None:
    filename = uploaded_file.name

    if not allowed_file(filename):
        st.error("❌ Invalid file type. Please upload an image or video.")
    else:
        # Save file temporarily
        with NamedTemporaryFile(delete=False, dir=UPLOAD_FOLDER, suffix=os.path.splitext(filename)[1]) as temp_file:
            temp_file.write(uploaded_file.read())
            filepath = temp_file.name

        st.success(f"✅ File uploaded: `{filename}`")

        try:
            if filename.lower().endswith(('mp4', 'avi', 'mov')):
                st.video(filepath)
                st.info("⏳ Extracting frames...")
                frames = extract_frames(filepath, NUM_FRAMES)

                if frames.size == 0:
                    st.error("⚠️ No frames could be extracted from the video.")
                else:
                    with st.spinner("🔍 Predicting deepfake probability..."):
                        predictions = model.predict(frames).flatten()
                        avg_prediction = np.mean(predictions)
                        file_type = 'video'
            else:
                st.image(filepath, caption="Uploaded Image", use_container_width=True)
                img_array = prepare_image(filepath)
                with st.spinner("🔍 Predicting deepfake probability..."):
                    avg_prediction = model.predict(img_array)[0][0]
                    file_type = 'image'

            # Interpretation
            if avg_prediction < FAKE_THRESHOLD:
                result = "🟥 Fake"
            elif avg_prediction > REAL_THRESHOLD:
                result = "🟩 Real"
            else:
                result = "🟨 Uncertain"

            st.markdown("---")
            st.subheader("📊 Prediction Result")
            st.write(f"**🧾 Result:** {result}")
            st.write(f"**📈 Confidence Score:** `{avg_prediction:.4f}`")
            st.write(f"**📁 File Type:** `{file_type}`")

        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
        finally:
            os.remove(filepath)
