import os
import io
import copy
import numpy as np
import tensorflow as tf
from PIL import Image
from utils.gradcam import make_gradcam_heatmap, overlay_gradcam

model_path = "outputs/models/melanoma_detector_model.h5"
model = tf.keras.models.load_model(model_path)

img_path = "dataset/test/melanoma/melanoma_10122.jpg"

# Simulate exactly what Streamlit does:
with open(img_path, "rb") as f:
    file_bytes = f.read()
    
uploaded_file = io.BytesIO(file_bytes)

# Direct Image Preprocessing
image = Image.open(uploaded_file).convert('RGB')
image_resized = image.resize((224, 224))
img_array = tf.keras.utils.img_to_array(image_resized)
img_tensor = tf.expand_dims(img_array, axis=0)

# Execute Prediction
prediction = model.predict(img_tensor, verbose=0)
prediction_prob = float(prediction[0][0]) 
print(f"Prediction Prob: {prediction_prob:.6f}")

# Execute Grad-CAM
heatmap = make_gradcam_heatmap(img_tensor, model)
gradcam_img = overlay_gradcam(img_tensor, heatmap)

print(f"Grad-CAM img shape: {gradcam_img.size}")
gradcam_img.save("test_gradcam_output.jpg")
