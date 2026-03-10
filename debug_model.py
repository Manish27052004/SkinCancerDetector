import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

model_path = "outputs/models/melanoma_detector_model.h5"
model = tf.keras.models.load_model(model_path)

test_dir = "dataset/test"

# Use image_dataset_from_directory to load the test set exactly like training
target_size = (224, 224)
batch_size = 32

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=target_size,
    batch_size=batch_size,
    shuffle=False # Keep order for accurate labels
)

print(f"\nClass Names: {test_dataset.class_names}")

print("\nEvaluating model on test dataset...")
# Get predictions and true labels
all_preds = []
all_labels = []

for images, labels in test_dataset:
    preds = model.predict(images, verbose=0)
    all_preds.extend(preds.flatten())
    all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Convert probabilities to binary classes
pred_classes = (all_preds > 0.5).astype(int)

print("\n--- Confusion Matrix ---")
print(confusion_matrix(all_labels, pred_classes))

print("\n--- Classification Report ---")
print(classification_report(all_labels, pred_classes, target_names=test_dataset.class_names))

print("\nAverage Prediction Probability for Benign (Class 0):", np.mean(all_preds[all_labels == 0]))
print("Average Prediction Probability for Melanoma (Class 1):", np.mean(all_preds[all_labels == 1]))
