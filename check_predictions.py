import os
import tensorflow as tf
import numpy as np
from PIL import Image

model_path = "outputs/models/melanoma_detector_model.h5"
model = tf.keras.models.load_model(model_path)

def predict_img(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.expand_dims(img, axis=0)
    return model.predict(img, verbose=0)[0][0]

def predict_pil(img_path):
    image = Image.open(img_path).convert('RGB')
    image_resized = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(image_resized)
    img_tensor = tf.expand_dims(img_array, axis=0)
    return model.predict(img_tensor, verbose=0)[0][0]

test_melanoma = os.listdir("dataset/test/melanoma")[:5]
test_benign = os.listdir("dataset/test/benign")[:5]

print("--- TF Preprocessing ---")
print("Melanoma predictions (should be close to 1.0):")
for f in test_melanoma:
    path = os.path.join("dataset/test/melanoma", f)
    print(f"{f}: {predict_img(path):.4f}")

print("\nBenign predictions (should be close to 0.0):")
for f in test_benign:
    path = os.path.join("dataset/test/benign", f)
    print(f"{f}: {predict_img(path):.4f}")

print("\n--- PIL Preprocessing (from app.py) ---")
print("Melanoma predictions (should be close to 1.0):")
for f in test_melanoma:
    path = os.path.join("dataset/test/melanoma", f)
    print(f"{f}: {predict_pil(path):.4f}")

print("\nBenign predictions (should be close to 0.0):")
for f in test_benign:
    path = os.path.join("dataset/test/benign", f)
    print(f"{f}: {predict_pil(path):.4f}")
