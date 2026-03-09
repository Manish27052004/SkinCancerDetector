"""
train_model.py

This wrapper script serves as the main entry point to train the model.
It integrates data preprocessing, model building, and training loops.
It also saves the best model to the outputs/ directory.
"""

import os
import sys
import tensorflow as tf

# Add the project root to the python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_builder import build_model
from utils.preprocessing import load_training_data, load_validation_data, load_test_data
from utils.metrics import plot_training_history, calculate_classification_metrics

def train():
    """
    Main sequence for model training.
    - Load augmented dataset.
    - Build/compile model.
    - Setup callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau).
    - Train and save the optimal model.
    """
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    test_dir = os.path.join(dataset_dir, 'test')
    
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
    models_dir = os.path.join(outputs_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, 'melanoma_detector_model.h5')

    batch_size = 32
    epochs = 25
    target_size = (224, 224)

    print("Loading datasets...")
    train_ds = load_training_data(train_dir, batch_size=batch_size, target_size=target_size)
    val_ds = load_validation_data(val_dir, batch_size=batch_size, target_size=target_size)

    print("Building model...")
    model = build_model(input_shape=(224, 224, 3), num_classes=1)

    print("Setting up callbacks...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    print(f"Training complete. Best model saved to {model_save_path}")
    
    # Plot history
    plot_training_history(history, outputs_dir)

    print("Evaluating on test dataset...")
    test_ds = load_test_data(test_dir, batch_size=batch_size, target_size=target_size)
    
    # Basic Evaluation
    results = model.evaluate(test_ds, verbose=0)
    print("\n--- Test Results ---")
    print(f"Loss: {results[0]:.4f}")
    print(f"Accuracy: {results[1]:.4f}")
    print(f"Precision: {results[2]:.4f}")
    print(f"Recall: {results[3]:.4f}")

    # Detailed metrics (F1, Confusion Matrix, ROC)
    print("\nCalculating detailed metrics...")
    
    # Extract true labels and predictions
    y_true = []
    y_pred_probs = []
    
    for images, labels in test_ds:
        y_true.extend(labels.numpy())
        preds = model.predict(images, verbose=0)
        y_pred_probs.extend(preds)
        
    calculate_classification_metrics(y_true, y_pred_probs, outputs_dir)


if __name__ == "__main__":
    train()
