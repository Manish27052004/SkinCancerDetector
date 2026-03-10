import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm

model_path = "outputs/models/melanoma_detector_model.h5"
model = tf.keras.models.load_model(model_path)

def make_gradcam_heatmap(img_tensor, model):
    base_model = model.layers[0]
    
    # Try to find the top activation
    last_conv_layer_name = 'top_activation'
    try:
        last_conv_layer = base_model.get_layer(last_conv_layer_name)
    except:
        last_conv_layer = [l for l in base_model.layers if len(l.output_shape) == 4][-1]

    # Create model to output conv features and full base model output
    base_model_extractor = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[last_conv_layer.output, base_model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, base_model_output = base_model_extractor(img_tensor)
        tape.watch(last_conv_layer_output)
        
        # Pass base model output through the rest of the sequential layers
        x = base_model_output
        for layer in model.layers[1:]:
            x = layer(x)
            
        preds = x # Single sigmoid probability
        class_channel = preds[:, 0]

    # Compute gradients
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    if grads is None:
        print("Error: Gradient is None")
        return np.zeros((last_conv_layer_output.shape[1], last_conv_layer_output.shape[2]))

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # ReLU
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Load test image
img_path = "dataset/test/melanoma/melanoma_10105.jpg"
image = Image.open(img_path).convert('RGB')
image_resized = image.resize((224, 224))
img_array = tf.keras.utils.img_to_array(image_resized)
img_tensor = tf.expand_dims(img_array, axis=0)

print("Testing Grad-CAM logic...")
heatmap = make_gradcam_heatmap(img_tensor, model)

print("Heatmap shape:", heatmap.shape)
print("Heatmap min/max:", np.min(heatmap), np.max(heatmap))
print("Grad-CAM extraction successful!")
