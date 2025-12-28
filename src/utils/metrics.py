"""Evaluation metrics and visualization."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import torch


def calculate_metrics(y_true, y_pred, y_prob=None, class_names=None):
    """
    Calculate classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        class_names: List of class names

    Returns:
        Dictionary of metrics
    """
    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch. Tensor):
        y_pred = y_pred.cpu().numpy()
    if y_prob is not None and isinstance(y_prob, torch. Tensor):
        y_prob = y_prob.cpu().numpy()

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "support_per_class": support,
        "precision_avg": precision_avg,
        "recall_avg":  recall_avg,
        "f1_avg": f1_avg,
    }

    # Calculate AUC if probabilities provided
    if y_prob is not None:
        try:
            # Multi-class AUC (one-vs-rest)
            n_classes = y_prob.shape[1]
            y_true_binary = np.eye(n_classes)[y_true]
            auc_per_class = []
            for i in range(n_classes):
                auc = roc_auc_score(y_true_binary[:, i], y_prob[:, i])
                auc_per_class.append(auc)
            metrics["auc_per_class"] = np.array(auc_per_class)
            metrics["auc_avg"] = np.mean(auc_per_class)
        except Exception as e:
            print(f"Could not calculate AUC: {e}")

    return metrics


def print_metrics(metrics, class_names=None):
    """
    Print metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics
        class_names: List of class names
    """
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Weighted Precision: {metrics['precision_avg']:.4f}")
    print(f"Weighted Recall: {metrics['recall_avg']:.4f}")
    print(f"Weighted F1-Score: {metrics['f1_avg']:.4f}")

    if "auc_avg" in metrics:
        print(f"Average AUC: {metrics['auc_avg']:.4f}")

    print("\n" + "-" * 60)
    print("PER-CLASS METRICS")
    print("-" * 60)

    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(metrics["precision_per_class"]))]

    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)

    for i, class_name in enumerate(class_names):
        print(
            f"{class_name:<15} "
            f"{metrics['precision_per_class'][i]: <12.4f} "
            f"{metrics['recall_per_class'][i]:<12.4f} "
            f"{metrics['f1_per_class'][i]:<12.4f} "
            f"{metrics['support_per_class'][i]:<10.0f}"
        )

        if "auc_per_class" in metrics:
            print(f"  AUC: {metrics['auc_per_class'][i]:.4f}")

    print("=" * 60 + "\n")


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """
    Plot confusion matrix.

    Args:
        y_true:  Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save figure
    """
    # Convert to numpy if tensors
    if isinstance(y_true, torch. Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_roc_curves(y_true, y_prob, class_names=None, save_path=None):
    """
    Plot ROC curves for multi-class classification.

    Args:
        y_true: Ground truth labels
        y_prob:  Predicted probabilities
        class_names: List of class names
        save_path: Path to save figure
    """
    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()

    n_classes = y_prob.shape[1]
    y_true_binary = np.eye(n_classes)[y_true]

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    # Plot ROC curve for each class
    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_prob[:, i])
        auc = roc_auc_score(y_true_binary[: , i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {auc:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves - Multi-Class Classification", fontsize=16, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"ROC curves saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and accuracy).

    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    ax1.plot(history["train_loss"], label="Train Loss", linewidth=2)
    ax1.plot(history["val_loss"], label="Validation Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Plot accuracy
    ax2.plot(history["train_acc"], label="Train Accuracy", linewidth=2)
    ax2.plot(history["val_acc"], label="Validation Accuracy", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training history saved to {save_path}")
    else:
        plt.show()

    plt.close()
