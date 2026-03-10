"""
preprocessing.py

This module contains utility functions for data loading, preprocessing,
and data augmentation. It prepares the raw images from the dataset folder
for training and evaluation.
"""

import tensorflow as tf
from tensorflow.keras import layers

def get_data_augmentation():
    """Returns a sequential model for data augmentation."""
    return tf.keras.Sequential([
        layers.RandomRotation(0.2),
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2)
    ])

def load_training_data(dataset_dir, batch_size=32, target_size=(224, 224)):
    """
    Loads and preprocesses the training dataset with data augmentation.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels='inferred',
        label_mode='binary',
        class_names=['benign', 'melanoma'],
        batch_size=batch_size,
        image_size=target_size,
        shuffle=True
    )
    
    # Apply data augmentation
    data_augmentation = get_data_augmentation()
    
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_ds

def load_validation_data(dataset_dir, batch_size=32, target_size=(224, 224)):
    """
    Loads and preprocesses the validation dataset without augmentation.
    """
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels='inferred',
        label_mode='binary',
        class_names=['benign', 'melanoma'],
        batch_size=batch_size,
        image_size=target_size,
        shuffle=False
    )
    
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return val_ds

def load_test_data(dataset_dir, batch_size=32, target_size=(224, 224)):
    """
    Loads and preprocesses the test dataset without augmentation.
    """
    test_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels='inferred',
        label_mode='binary',
        class_names=['benign', 'melanoma'],
        batch_size=batch_size,
        image_size=target_size,
        shuffle=False
    )
    
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return test_ds

def preprocess_single_image(image_path, target_size=(224, 224)):
    """
    Preprocesses a single image for model prediction.
    Reads the image, resizes, and normalizes it.
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    
    # Add a batch dimension (e.g., from (224, 224, 3) to (1, 224, 224, 3))
    return tf.expand_dims(img, axis=0)
