"""Quick confusion matrix from validation results."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import argparse


def plot_cm(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    # Add annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)'
            color = 'white' if cm[i,j] > cm.max()/2 else 'black'
            ax.text(j+0.5, i+0.5, text, ha='center', va='center', 
                   color=color, fontsize=13, fontweight='bold')
    
    ax.set_title('Confusion Matrix', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to: {save_path}")


if __name__ == "__main__": 
    # Example:  Load from CSV or numpy arrays
    # Replace with your actual data
    
    # Option 1: From numpy arrays
    # y_true = np.load('y_true.npy')
    # y_pred = np.load('y_pred.npy')
    
    # Option 2: From CSV
    # import pandas as pd
    # df = pd.read_csv('validation_results.csv')
    # y_true = df['true_label'].values
    # y_pred = df['predicted_label'].values
    
    # Class names
    class_names = ['normal', 'crackle', 'wheeze', 'both']
    
    # Generate plot
    # plot_cm(y_true, y_pred, class_names)
    
    print("Update this script with your actual validation data!")