"""ICBHI 2017 Challenge scoring metrics."""

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_sensitivity_specificity(y_true, y_pred, class_idx):
    """
    Calculate sensitivity and specificity for a specific class.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_idx: Class index to calculate metrics for
    
    Returns:
        Tuple of (sensitivity, specificity)
    """
    # Convert to binary:  class_idx vs rest
    y_true_binary = (y_true == class_idx).astype(int)
    y_pred_binary = (y_pred == class_idx).astype(int)
    
    # Calculate confusion matrix components
    TP = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    TN = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    FP = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    FN = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    
    # Sensitivity (Recall, True Positive Rate)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    # Specificity (True Negative Rate)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    
    return sensitivity, specificity


def calculate_icbhi_score(y_true, y_pred, class_names=None):
    """
    Calculate ICBHI 2017 Challenge score.
    
    The ICBHI score is the average of sensitivity and specificity
    across all classes (harmonic mean approach).
    
    For respiratory sounds: 
    - Normal (0)
    - Crackle (1)
    - Wheeze (2)
    - Both (3)
    
    Args:
        y_true: Ground truth labels (numpy array or list)
        y_pred: Predicted labels (numpy array or list)
        class_names: List of class names (optional)
    
    Returns:
        Dictionary containing ICBHI score and detailed metrics
    """
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_pred, list):
        y_pred = np. array(y_pred)
    
    if class_names is None:
        class_names = ['normal', 'crackle', 'wheeze', 'both']
    
    num_classes = len(class_names)
    
    # Calculate sensitivity and specificity for each class
    sensitivities = []
    specificities = []
    
    metrics_per_class = {}
    
    for class_idx, class_name in enumerate(class_names):
        sens, spec = calculate_sensitivity_specificity(y_true, y_pred, class_idx)
        sensitivities.append(sens)
        specificities.append(spec)
        
        # Harmonic mean (F1-like score)
        if sens + spec > 0:
            hs = 2 * (sens * spec) / (sens + spec)
        else:
            hs = 0.0
        
        metrics_per_class[class_name] = {
            'sensitivity': sens,
            'specificity': spec,
            'harmonic_score': hs
        }
    
    # Calculate overall ICBHI score
    # Method 1: Average of all sensitivities and specificities
    avg_sensitivity = np.mean(sensitivities)
    avg_specificity = np.mean(specificities)
    
    # ICBHI Score:  Harmonic mean of average sensitivity and specificity
    if avg_sensitivity + avg_specificity > 0:
        icbhi_score = 2 * (avg_sensitivity * avg_specificity) / (avg_sensitivity + avg_specificity)
    else:
        icbhi_score = 0.0
    
    # Method 2: Average of harmonic scores per class
    avg_harmonic_score = np.mean([m['harmonic_score'] for m in metrics_per_class.values()])
    
    # Calculate overall accuracy
    accuracy = np.mean(y_true == y_pred)
    
    results = {
        'icbhi_score': icbhi_score,
        'avg_sensitivity': avg_sensitivity,
        'avg_specificity': avg_specificity,
        'avg_harmonic_score': avg_harmonic_score,
        'accuracy': accuracy,
        'per_class_metrics': metrics_per_class,
        'sensitivities': sensitivities,
        'specificities': specificities
    }
    
    return results


def print_icbhi_metrics(metrics, class_names=None):
    """
    Print ICBHI metrics in a formatted way.
    
    Args:
        metrics: Dictionary returned from calculate_icbhi_score
        class_names: List of class names
    """
    if class_names is None:
        class_names = ['normal', 'crackle', 'wheeze', 'both']
    
    print("\n" + "=" * 70)
    print("ICBHI 2017 CHALLENGE SCORE")
    print("=" * 70)
    
    # Overall metrics
    print(f"\n{'Metric':<30} {'Value': >10}")
    print("-" * 70)
    print(f"{'ICBHI Score (Main Metric)':<30} {metrics['icbhi_score']: >10.4f}")
    print(f"{'Average Sensitivity':<30} {metrics['avg_sensitivity']:>10.4f}")
    print(f"{'Average Specificity':<30} {metrics['avg_specificity']: >10.4f}")
    print(f"{'Average Harmonic Score':<30} {metrics['avg_harmonic_score']: >10.4f}")
    print(f"{'Overall Accuracy':<30} {metrics['accuracy']:>10.4f}")
    
    # Per-class metrics
    print("\n" + "-" * 70)
    print("PER-CLASS METRICS")
    print("-" * 70)
    print(f"{'Class':<15} {'Sensitivity':<15} {'Specificity':<15} {'H-Score':<15}")
    print("-" * 70)
    
    for class_name in class_names:
        class_metrics = metrics['per_class_metrics'][class_name]
        print(
            f"{class_name:<15} "
            f"{class_metrics['sensitivity']:<15.4f} "
            f"{class_metrics['specificity']: <15.4f} "
            f"{class_metrics['harmonic_score']:<15.4f}"
        )
    
    print("=" * 70 + "\n")


def plot_icbhi_metrics(metrics, class_names=None, save_path=None):
    """
    Visualize ICBHI metrics. 
    
    Args:
        metrics: Dictionary returned from calculate_icbhi_score
        class_names: List of class names
        save_path: Path to save the plot
    """
    if class_names is None:
        class_names = ['normal', 'crackle', 'wheeze', 'both']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Sensitivity and Specificity per class
    ax1 = axes[0]
    x = np.arange(len(class_names))
    width = 0.35
    
    sensitivities = [metrics['per_class_metrics'][c]['sensitivity'] for c in class_names]
    specificities = [metrics['per_class_metrics'][c]['specificity'] for c in class_names]
    
    bars1 = ax1.bar(x - width/2, sensitivities, width, label='Sensitivity', alpha=0.8)
    bars2 = ax1.bar(x + width/2, specificities, width, label='Specificity', alpha=0.8)
    
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Sensitivity and Specificity per Class', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names)
    ax1.legend()
    ax1.set_ylim([0, 1.0])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]: 
        for bar in bars: 
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Overall metrics comparison
    ax2 = axes[1]
    overall_metrics = {
        'ICBHI Score': metrics['icbhi_score'],
        'Avg Sensitivity': metrics['avg_sensitivity'],
        'Avg Specificity': metrics['avg_specificity'],
        'Accuracy': metrics['accuracy']
    }
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    bars = ax2.barh(list(overall_metrics.keys()), list(overall_metrics.values()), color=colors, alpha=0.8)
    
    ax2.set_xlabel('Score', fontsize=12)
    ax2.set_title('Overall Metrics', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 1.0])
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.4f}',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path: 
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ICBHI metrics plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def calculate_detailed_confusion_metrics(y_true, y_pred, class_names=None):
    """
    Calculate detailed confusion matrix metrics for all classes.
    
    Args:
        y_true: Ground truth labels
        y_pred:  Predicted labels
        class_names:  List of class names
    
    Returns:
        Dictionary with detailed metrics
    """
    if class_names is None:
        class_names = ['normal', 'crackle', 'wheeze', 'both']
    
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {}
    
    for idx, class_name in enumerate(class_names):
        TP = cm[idx, idx]
        FP = cm[: , idx].sum() - TP
        FN = cm[idx, :].sum() - TP
        TN = cm. sum() - TP - FP - FN
        
        # Calculate metrics
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        f1_score = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
        
        metrics[class_name] = {
            'TP':  int(TP),
            'FP': int(FP),
            'FN': int(FN),
            'TN': int(TN),
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1_score
        }
    
    return metrics, cm


def plot_detailed_confusion_matrix(cm, class_names=None, save_path=None):
    """
    Plot confusion matrix with detailed annotations.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
    """
    if class_names is None: 
        class_names = ['normal', 'crackle', 'wheeze', 'both']
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure
    fig, ax = plt. subplots(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax)
    
    # Add custom annotations (count + percentage)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
            color = 'white' if cm[i, j] > cm. max() / 2 else 'black'
            ax.text(j + 0.5, i + 0.5, text,
                   ha='center', va='center', color=color, fontsize=11, fontweight='bold')
    
    ax.set_title('Confusion Matrix (Count and Percentage)', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=13)
    ax.set_xlabel('Predicted Label', fontsize=13)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Detailed confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__": 
    # Test the metrics
    print("Testing ICBHI metrics.. .\n")
    
    # Simulated predictions
    np.random.seed(42)
    y_true = np.random.randint(0, 4, 100)
    y_pred = np.random.randint(0, 4, 100)
    
    # Calculate ICBHI score
    metrics = calculate_icbhi_score(y_true, y_pred)
    
    # Print metrics
    print_icbhi_metrics(metrics)
    
    # Plot metrics
    plot_icbhi_metrics(metrics, save_path='test_icbhi_metrics.png')
    
    # Detailed confusion metrics
    detailed_metrics, cm = calculate_detailed_confusion_metrics(y_true, y_pred)
    
    print("\nDetailed Confusion Matrix Metrics:")
    print("-" * 70)
    for class_name, class_metrics in detailed_metrics.items():
        print(f"\n{class_name. upper()}:")
        print(f"  TP: {class_metrics['TP']}, FP: {class_metrics['FP']}, "
              f"FN: {class_metrics['FN']}, TN: {class_metrics['TN']}")
        print(f"  Precision: {class_metrics['precision']:.4f}")
        print(f"  F1-Score: {class_metrics['f1_score']:.4f}")
    
    plot_detailed_confusion_matrix(cm, save_path='test_confusion_matrix.png')
