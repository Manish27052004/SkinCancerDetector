"""
predict_image.py

This script allows users to load a trained model and pass a new, unseen
skin lesion image to it. It returns the predicted class (Benign or Melanoma)
along with the confidence score.
"""

import sys
import os
import argparse
import tensorflow as tf

# Add the project root to the python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import preprocess_single_image

def predict(image_path, model_path):
    """
    Predicts whether a skin lesion is Benign or Melanoma.
    
    Args:
        image_path (str): Path to the image file to classify.
        model_path (str): Path to the saved trained model (.h5 or .keras).
        
    Returns:
        dict: A dictionary containing the prediction label and confidence score.
    """
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    
    # Preprocess the input image
    # The function resizes to 224x224, normalizes (0-1), and adds a batch dimension
    img_tensor = preprocess_single_image(image_path, target_size=(224, 224))
    
    # Perform inference using the model
    prediction_prob = model.predict(img_tensor, verbose=0)[0][0]
    
    # Map model output (probability) to class label (Benign or Melanoma)
    # The output is a probability of being 'Melanoma' (since class 1 is Melanoma)
    if prediction_prob > 0.5:
        prediction_label = "Melanoma"
        confidence = prediction_prob
    else:
        prediction_label = "Benign"
        confidence = 1.0 - prediction_prob
        
    print(f"\nPrediction: {prediction_label}")
    print(f"Confidence: {confidence:.2f}")
    
    return {
        'prediction': prediction_label,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Melanoma from a skin lesion image.")
    parser.add_argument("image_path", help="Path to the custom image to predict (must be provided).")
    
    # Determine default model path dynamically
    default_model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        "outputs", 
        "models", 
        "melanoma_detector_model.h5"
    )
    
    parser.add_argument("--model", type=str, default=default_model_path,
                        help="Path to the trained model (.h5 or .keras).")
                        
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at '{args.image_path}'")
        sys.exit(1)
        
    if not os.path.exists(args.model):
        print(f"Error: Model not found at '{args.model}'")
        sys.exit(1)
        
    # Run the prediction
    predict(args.image_path, args.model)
