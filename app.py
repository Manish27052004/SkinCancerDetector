import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import sys
import tempfile

# Add the project root to the python path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.preprocessing import preprocess_single_image

# --- Configuration ---
st.set_page_config(
    page_title="Melanoma Skin Cancer AI",
    page_icon="🧬",
    layout="centered"
)

# --- Advanced Custom UI (CSS) ---
# Injecting a psychological color palette for medical trust (deep blues, clean teals, and soft darks).
# Utilizing glassmorphism and smooth animations.
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background-color: #0b132b;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1 {
        color: #48cae4 !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
        text-align: center;
        margin-bottom: 0px !important;
    }
    h2, h3 {
        color: #90e0ef !important;
        font-weight: 500 !important;
    }
    
    /* Subtitle text */
    .subtitle {
        text-align: center;
        color: #caf0f8;
        font-size: 1.1rem;
        margin-bottom: 30px;
        opacity: 0.8;
    }
    
    /* File Uploader Customization */
    .stFileUploader > div > div {
        background: rgba(72, 202, 228, 0.05);
        border: 2px dashed #48cae4;
        border-radius: 16px;
        padding: 30px;
        transition: all 0.3s ease;
    }
    .stFileUploader > div > div:hover {
        background: rgba(72, 202, 228, 0.1);
        border-color: #90e0ef;
    }

    /* Primary Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #0077b6, #00b4d8);
        color: white !important;
        font-size: 1.1rem;
        font-weight: 700;
        border-radius: 30px;
        border: none;
        padding: 12px 28px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 180, 216, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 180, 216, 0.5);
    }
    
    /* Image container */
    [data-testid="stImage"] {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* Result Cards */
    .result-card-melanoma {
        background: rgba(239, 35, 60, 0.1);
        border-left: 6px solid #ef233c;
        padding: 24px;
        border-radius: 12px;
        margin-top: 25px;
        backdrop-filter: blur(10px);
    }
    .result-card-benign {
        background: rgba(43, 147, 72, 0.1);
        border-left: 6px solid #2b9348;
        padding: 24px;
        border-radius: 12px;
        margin-top: 25px;
        backdrop-filter: blur(10px);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_trained_model():
    """Loads the Keras model. Cached to prevent reloading on every interaction."""
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "models", "melanoma_detector_model.h5")
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

# --- Main App ---
def main():
    st.markdown("<h1>🧬 AI Dermoscopy System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Advanced Neural Network for Skin Lesion Classification</p>", unsafe_allow_html=True)

    # Check Model Status
    model = load_trained_model()
    if model is None:
        st.warning("⚠️ **System Offline:** Neural Network weights not found. Please train the model using `train_model.py` before analyzing images.")
        st.stop() # Prevents the rest of the app from running

    # File Uploader
    uploaded_file = st.file_uploader("Upload a High-Resolution Dermoscopic Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Layout columns for a sleek centered display
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            
            # Predict Button
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_button = st.button("🔍 Analyze Lesion")

        if analyze_button:
            with st.spinner("AI is analyzing microscopic cellular features..."):
                try:
                    # Save temp file for preprocessing
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                        temp_file.write(uploaded_file.getbuffer())
                        temp_img_path = temp_file.name
                        
                    # Preprocess and Predict
                    img_tensor = preprocess_single_image(temp_img_path, target_size=(224, 224))
                    prediction_prob = model.predict(img_tensor, verbose=0)[0][0]
                    
                    # Interpret results
                    if prediction_prob > 0.5:
                        prediction_label = "Melanoma Detected"
                        confidence = prediction_prob
                        
                        st.markdown(f"""
                        <div class="result-card-melanoma">
                            <h2 style="color: #ef233c !important; margin: 0;">⚠️ {prediction_label}</h2>
                            <p style="font-size: 1.1rem; margin: 8px 0 0 0; color: #edf2f4;">The AI has identified characteristics highly consistent with Melanoma.</p>
                            <h3 style="margin-top: 15px; color: #ef233c !important; font-size: 1.5rem;">Confidence Score: {(confidence*100):.1f}%</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(float(confidence))
                    else:
                        prediction_label = "Benign Lesion"
                        confidence = 1.0 - prediction_prob
                        
                        st.markdown(f"""
                        <div class="result-card-benign">
                            <h2 style="color: #2b9348 !important; margin: 0;">✅ {prediction_label}</h2>
                            <p style="font-size: 1.1rem; margin: 8px 0 0 0; color: #edf2f4;">The AI did not find significant characteristics of Melanoma.</p>
                            <h3 style="margin-top: 15px; color: #2b9348 !important; font-size: 1.5rem;">Confidence Score: {(confidence*100):.1f}%</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(float(confidence))
                        
                except Exception as e:
                    st.error(f"An anomaly occurred during neural processing: {str(e)}")
                finally:
                    # Cleanup
                    if 'temp_img_path' in locals() and os.path.exists(temp_img_path):
                        os.remove(temp_img_path)

if __name__ == "__main__":
    main()
