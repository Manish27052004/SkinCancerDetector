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
from utils.gradcam import make_gradcam_heatmap, overlay_gradcam

# --- Configuration ---
st.set_page_config(
    page_title="Melanoma Skin Cancer AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Advanced Custom UI (CSS) ---
st.markdown("""
<style>
    /* Global Background and Typography */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700;900&display=swap');
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stApp {
        background: linear-gradient(-45deg, #0b132b, #1c2541, #0f1c3f, #070d1e);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #e0e1dd;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header Container */
    .header-container {
        padding: 50px 0;
        text-align: center;
        background: radial-gradient(circle at 50% 100%, rgba(72, 202, 228, 0.1) 0%, rgba(13, 27, 42, 0) 70%);
        border-bottom: 1px solid rgba(119, 141, 169, 0.15);
        margin-bottom: 50px;
        border-radius: 0 0 40px 40px;
    }
    
    /* Gradient Title */
    h1 {
        background: linear-gradient(90deg, #90e0ef 0%, #00b4d8 50%, #0077b6 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        letter-spacing: -2px;
        margin-bottom: 10px !important;
        font-size: 4.2rem !important;
        animation: shine 5s linear infinite;
    }
    
    @keyframes shine {
        to { background-position: 200% center; }
    }
    
    /* Subtitle text */
    .subtitle {
        color: #a3bac3;
        font-size: 1.3rem;
        font-weight: 300;
        margin-top: 5px;
        letter-spacing: 0.5px;
    }
    
    /* Panel Glassmorphism */
    .glass-panel {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 24px;
        padding: 35px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px) saturate(180%);
        -webkit-backdrop-filter: blur(12px) saturate(180%);
    }

    /* File Uploader Customization */
    .stFileUploader > div > div {
        background: rgba(0, 180, 216, 0.02) !important;
        border: 2px dashed rgba(0, 180, 216, 0.5) !important;
        border-radius: 20px;
        padding: 50px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .stFileUploader > div > div:hover {
        background: rgba(0, 180, 216, 0.05) !important;
        border-color: #00b4d8 !important;
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 10px 30px rgba(0, 180, 216, 0.15);
    }

    /* Primary Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #0077b6, #00b4d8);
        color: white !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        border-radius: 16px;
        border: none;
        padding: 15px 30px !important;
        width: 100%;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 180, 216, 0.2);
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 180, 216, 0.4);
        background: linear-gradient(135deg, #0096c7, #48cae4);
    }
    
    /* Image container */
    [data-testid="stImage"] {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 0 auto;
        display: flex;
        justify-content: center;
        transition: transform 0.3s ease;
    }
    [data-testid="stImage"]:hover {
        transform: scale(1.02);
    }

    /* Animated Result Cards */
    @keyframes slideUpFade {
        from { opacity: 0; transform: translateY(30px) scale(0.95); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes pulseShadowMelanoma {
        0% { box-shadow: 0 0 0 0 rgba(208, 0, 0, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(208, 0, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(208, 0, 0, 0); }
    }
    @keyframes pulseShadowBenign {
        0% { box-shadow: 0 0 0 0 rgba(56, 176, 0, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(56, 176, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(56, 176, 0, 0); }
    }
    
    .result-card-melanoma {
        background: linear-gradient(135deg, rgba(208, 0, 0, 0.2) 0%, rgba(157, 2, 8, 0.05) 100%);
        border: 1px solid rgba(208, 0, 0, 0.4);
        border-left: 8px solid #ff0000;
        padding: 35px;
        border-radius: 20px;
        margin-top: 25px;
        backdrop-filter: blur(12px);
        animation: slideUpFade 0.7s cubic-bezier(0.16, 1, 0.3, 1), pulseShadowMelanoma 2s infinite;
    }
    
    .result-card-benign {
        background: linear-gradient(135deg, rgba(56, 176, 0, 0.2) 0%, rgba(0, 128, 0, 0.05) 100%);
        border: 1px solid rgba(56, 176, 0, 0.4);
        border-left: 8px solid #38b000;
        padding: 35px;
        border-radius: 20px;
        margin-top: 25px;
        backdrop-filter: blur(12px);
        animation: slideUpFade 0.7s cubic-bezier(0.16, 1, 0.3, 1), pulseShadowBenign 2s infinite;
    }
    
    .result-title {
        margin: 0 0 10px 0;
        font-size: 2rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .result-desc {
        font-size: 1.1rem;
        color: #e0e1dd;
        margin: 0 0 20px 0;
        line-height: 1.5;
        opacity: 0.9;
    }
    
    .confidence-metric {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
    # Header Section
    st.markdown("""
    <div class="header-container">
        <h1>🧬 AI Skin Lesion Screen</h1>
        <p class="subtitle">Melanoma Detection Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Information
    with st.sidebar:
        st.markdown("### ℹ️ About the System")
        st.info(
            "This application uses an AI deep learning model trained **specifically to detect Melanoma skin cancer**."
        )
        st.markdown("### ⚠️ Important Limitations")
        st.warning(
            "1. **Melanoma Only:** This AI does **NOT** detect other skin conditions (like fungal infections, rashes, or other types of skin cancer). If you upload a picture of a fungal infection, the AI will just say 'Not Melanoma'.\n"
            "2. **Not a Doctor:** This is an educational tool. Always consult a dermatologist for medical advice."
        )
        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.markdown(
            "1. Upload a clear, zoomed-in picture of a single skin mole or lesion.\n"
            "2. The AI will **automatically** analyze the image as soon as you upload it.\n"
            "3. Review the confidence score below."
        )

    # Check Model Status
    model = load_trained_model()
    if model is None:
        st.error("⚠️ **System Offline:** Neural Network weights not found. Please train the model using `train_model.py` before analyzing images.")
        st.stop()

    # Main Content Area
    col1, col2, col3 = st.columns([1, 10, 1])
    
    with col2:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        # File Uploader
        uploaded_file = st.file_uploader("Upload Skin Image Here", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            
            try:
                # Direct Image Preprocessing
                image = Image.open(uploaded_file).convert('RGB')
                image_resized = image.resize((224, 224))
                img_array = tf.keras.utils.img_to_array(image_resized)
                img_tensor = tf.expand_dims(img_array, axis=0)
                
                with st.spinner("AI is examining the cellular structures..."):
                    # Execute Prediction
                    prediction = model.predict(img_tensor, verbose=0)
                    prediction_prob = float(prediction[0][0]) 
                    
                    # Execute Grad-CAM Explainable AI
                    heatmap = make_gradcam_heatmap(img_tensor, model)
                    gradcam_img = overlay_gradcam(image, heatmap)
                    
                # Create a 3-column layout for Original Image, Grad-CAM, and textual Results
                img_col, grad_col, res_col = st.columns([1, 1, 1.2], gap="large")
                
                with img_col:
                    st.markdown("<h4 style='text-align: center; color: #778da9; font-size: 1.1rem; margin-bottom: 15px;'>Original Image</h4>", unsafe_allow_html=True)
                    st.image(image, use_container_width=True)
                    
                with grad_col:
                    st.markdown("<h4 style='text-align: center; color: #778da9; font-size: 1.1rem; margin-bottom: 15px;'>AI Focus Map</h4>", unsafe_allow_html=True)
                    st.image(gradcam_img, use_container_width=True)

                with res_col:
                    # Prepare result aesthetics
                    if prediction_prob > 0.5:
                        label = "High Risk of Melanoma"
                        desc_text = "The AI detected patterns that look like **Melanoma** (a serious type of skin cancer). Please see a doctor immediately."
                        color_hex = "#d00000"
                        css_class = "result-card-melanoma"
                        icon = "⚠️"
                        conf = prediction_prob
                    else:
                        label = "Low Risk / Not Melanoma"
                        desc_text = "The AI did **not** see signs of Melanoma. <br><br><i>Note: This could still be a normal mole or a completely different skin issue (like a fungal infection). It just doesn't look like Melanoma to the AI.</i>"
                        color_hex = "#38b000"
                        css_class = "result-card-benign"
                        icon = "✅"
                        conf = 1.0 - prediction_prob
                        
                    # Display Result Card
                    st.markdown(f"""
                    <div class="{css_class}">
                        <h2 class="result-title" style="color: {color_hex} !important;">{icon} {label}</h2>
                        <p class="result-desc">{desc_text}</p>
                        <p style="margin:0; font-size:1rem; color:#778da9;">AI Confidence Level:</p>
                        <h3 class="confidence-metric" style="color: {color_hex} !important;">{(conf * 100):.1f}%</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Custom progress bar
                    st.progress(conf)
                        
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
