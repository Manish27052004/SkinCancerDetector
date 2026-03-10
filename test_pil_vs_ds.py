import tensorflow as tf
from PIL import Image
import numpy as np

model_path = "outputs/models/melanoma_detector_model.h5"
model = tf.keras.models.load_model(model_path)

img_path = "dataset/test/melanoma/melanoma_10105.jpg" # Known melanoma

print("\n--- 1. app.py PIL Method (Current App Logic) ---")
image = Image.open(img_path).convert('RGB')
image_resized = image.resize((224, 224))
img_array = tf.keras.utils.img_to_array(image_resized)
img_tensor = tf.expand_dims(img_array, axis=0)

print("PIL Shape:", img_tensor.shape)
print("PIL Min/Max:", np.min(img_tensor), np.max(img_tensor))
print("PIL Mean:", np.mean(img_tensor))
pred_pil = model.predict(img_tensor, verbose=0)
print("PIL Prediction:", pred_pil[0][0])


print("\n--- 2. tf.image Method (Used in dataset loading via TF) ---")
img_tf = tf.io.read_file(img_path)
img_tf = tf.image.decode_jpeg(img_tf, channels=3)
img_tf = tf.image.resize(img_tf, (224, 224))
img_tf_tensor = tf.expand_dims(img_tf, axis=0)

print("TF Shape:", img_tf_tensor.shape)
print("TF Min/Max:", np.min(img_tf_tensor), np.max(img_tf_tensor))
print("TF Mean:", np.mean(img_tf_tensor))
pred_tf = model.predict(img_tf_tensor, verbose=0)
print("TF Prediction:", pred_tf[0][0])
