"""
metrics.py

This module defines custom metrics, callbacks, and visualization tools
(e.g., confusion matrix, precision, recall, F1-score) to evaluate
the model's performance during and after training.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_curve, auc, ConfusionMatrixDisplay

def plot_training_history(history, outputs_dir):
    """
    Plots the training and validation accuracy/loss curves and saves them.
    """
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plot_path = os.path.join(outputs_dir, 'training_history.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Training history plot saved to {plot_path}")

def calculate_classification_metrics(y_true, y_pred_probs, outputs_dir):
    """
    Calculates detailed metrics such as Precision, Recall, F1-Score, AUC,
    and plots Confusion Matrix and ROC curve.
    """
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs).flatten()
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    # Calculate F1-Score
    f1 = f1_score(y_true, y_pred)
    print(f"F1 Score: {f1:.4f}")
    
    # Print full classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Benign', 'Melanoma']))
    
    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Melanoma'])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title('Confusion Matrix')
    cm_path = os.path.join(outputs_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")
    
    # Generate ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    roc_path = os.path.join(outputs_dir, 'roc_curve.png')
    plt.savefig(roc_path)
    plt.close()
    print(f"ROC curve saved to {roc_path}")
