import numpy as np
import tensorflow as tf
from PIL import Image
from utils.gradcam import make_gradcam_heatmap
import copy

model_path = "outputs/models/melanoma_detector_model.h5"
model = tf.keras.models.load_model(model_path)

img_path = "dataset/test/melanoma/melanoma_10105.jpg" # Known melanoma

print("--- Checking Prediction Continuity ---")
image = Image.open(img_path).convert('RGB')
image_resized = image.resize((224, 224))
img_array = tf.keras.utils.img_to_array(image_resized)
img_tensor = tf.expand_dims(img_array, axis=0)

# Prediction 1 (Pre-GradCAM)
pred_1 = model.predict(img_tensor, verbose=0)
print(f"Prediction 1 (Clean State): {pred_1[0][0]:.6f}")

# Run Grad-CAM
print("Running Grad-CAM...")
heatmap = make_gradcam_heatmap(img_tensor, model)

# Prediction 2 (Post-GradCAM)
pred_2 = model.predict(img_tensor, verbose=0)
print(f"Prediction 2 (After Grad-CAM): {pred_2[0][0]:.6f}")

if not np.isclose(pred_1[0][0], pred_2[0][0]):
    print("Warning! Model state was altered during Grad-CAM!")
else:
    print("Model state is safe.")
