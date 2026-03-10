import os
import tensorflow as tf
from PIL import Image

model_path = "outputs/models/melanoma_detector_model.h5"
model = tf.keras.models.load_model(model_path)

def predict_pil(img_path):
    image = Image.open(img_path).convert('RGB')
    image_resized = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(image_resized)
    img_tensor = tf.expand_dims(img_array, axis=0)
    return float(model.predict(img_tensor, verbose=0)[0][0])

melanoma_dir = "dataset/test/melanoma"
test_melanoma = os.listdir(melanoma_dir)

print("Finding highly confident melanoma images...")
found = 0
for f in test_melanoma:
    path = os.path.join(melanoma_dir, f)
    prob = predict_pil(path)
    if prob > 0.90:
        print(f"Confidence {prob*100:.2f}%: {path}")
        found += 1
        if found >= 5:
            break
