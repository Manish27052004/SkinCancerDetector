import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
from PIL import Image

def make_gradcam_heatmap(img_tensor, model):
    """
    Generates a Grad-CAM heatmap for a given image tensor and model.
    The model is expected to be a Sequential model where the first layer 
    is a base CNN (like EfficientNetB0).
    """
    # 1. Extract the base convolutional model
    base_model = model.layers[0]
    
    # 2. Find the last convolutional layer (for EfficientNetB0, it's typically 'top_activation')
    last_conv_layer_name = 'top_activation'
    try:
        last_conv_layer = base_model.get_layer(last_conv_layer_name)
    except ValueError:
        # Fallback dynamically if name doesn't exist
        last_conv_layer = [l for l in base_model.layers if len(l.output_shape) == 4][-1]

    # 3. Create a sub-model that outputs the conv features and the base model's final features
    base_model_extractor = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[last_conv_layer.output, base_model.output]
    )

    # 4. Compute gradients using tf.GradientTape
    with tf.GradientTape() as tape:
        last_conv_layer_output, base_model_output = base_model_extractor(img_tensor)
        # Watch the convolutional feature map to compute gradients with respect to it
        tape.watch(last_conv_layer_output)
        
        # Manually pass the base_model_output through the rest of the full model
        # (e.g., GlobalAveragePooling2D -> Dense)
        x = base_model_output
        for layer in model.layers[1:]:
            x = layer(x)
            
        preds = x # This is the final sigmoid confidence score (0.0 to 1.0)
        # We want the gradient of the class score (Melanoma)
        class_channel = preds[:, 0]

    # 5. Extract gradients of the top class score with respect to the feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Safety fallback if tracing fails
    if grads is None:
        return np.zeros((last_conv_layer_output.shape[1], last_conv_layer_output.shape[2]))

    # 6. Global average pooling of the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # 7. Multiply each feature map channel by its importance (pooled gradient)
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # 8. Apply ReLU and normalize
    # ReLU ensures we only highlight features that positively correlate with the target class
    heatmap = tf.maximum(heatmap, 0) 
    max_heat = tf.math.reduce_max(heatmap)
    if max_heat != 0:
        heatmap /= max_heat
        
    # --- Advanced Mathematical Sharpening ---
    # Apply a quadratic curve to suppress low-confidence background noise
    # and aggressively highlight the most critical pixels.
    heatmap = heatmap ** 2 
    
    return heatmap.numpy()


def overlay_gradcam(original_image, heatmap, alpha=0.4):
    """
    Overlays the Grad-CAM heatmap onto the original RGB image.
    Uses PIL and Matplotlib to maintain compatibility without requiring OpenCV.
    """
    # Convert PIL Image directly to numpy array to preserve original resolution & aspect ratio
    img_array = np.array(original_image)
    
    # Rescale heatmap to 0-255 range
    heatmap = np.uint8(255 * heatmap)

    # Use 'jet' colormap
    jet = cm.get_cmap("jet")

    # Access RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Resize jet heatmap to match original image shape exactly
    # Use BICUBIC interpolation for much smoother upscaling from the tiny 7x7 feature map
    jet_heatmap = tf.image.resize(
        jet_heatmap, 
        (img_array.shape[0], img_array.shape[1]),
        method=tf.image.ResizeMethod.BICUBIC
    ).numpy()

    # Superimpose the heatmap on original image using the alpha factor
    superimposed_img = jet_heatmap * alpha * 255 + img_array
    
    # Clip values to valid [0, 255] range
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return Image.fromarray(superimposed_img)
