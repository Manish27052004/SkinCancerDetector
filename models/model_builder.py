"""
model_builder.py

This module is responsible for defining the deep learning model architecture.
It includes functions to build and compile the neural network (e.g., using a pre-trained
model like ResNet50 or EfficientNet) suitable for binary classification (Benign vs. Melanoma).
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape=(224, 224, 3), num_classes=1):
    """
    Builds the CNN classification model using EfficientNetB0.
    
    Args:
        input_shape (tuple): The expected shape of input images.
        num_classes (int): Number of output units (1 for binary classification).
        
    Returns:
        tf.keras.Model: A compiled Keras model.
    """
    # Load the pre-trained EfficientNetB0 model
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, 
        weights='imagenet', 
        input_shape=input_shape
    )
    
    # Freeze the base model layers (optional, but good for initial training)
    # base_model.trainable = False

    # Create the model architecture
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_classes, activation='sigmoid')
    ])

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model
