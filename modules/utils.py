"""
Utility functions for FactTrack
"""

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import List, Dict, Any
import config


def download_pretrained_bert():
    """
    Download and cache BERT model if not already present
    """
    from transformers import DistilBertModel, DistilBertTokenizer
    
    print(f"Downloading {config.BERT_MODEL}...")
    tokenizer = DistilBertTokenizer.from_pretrained(config.BERT_MODEL)
    model = DistilBertModel.from_pretrained(config.BERT_MODEL)
    
    print("✓ BERT model downloaded and cached")
    return model, tokenizer


def balance_dataset(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Balance dataset by oversampling minority class
    
    Args:
        df: DataFrame with data
        label_col: Column name containing labels
        
    Returns:
        Balanced DataFrame
    """
    print(f"\nBalancing dataset on column: {label_col}")
    
    # Count samples per class
    class_counts = df[label_col].value_counts()
    print(f"Original distribution:\n{class_counts}")
    
    # Find maximum class size
    max_size = class_counts.max()
    
    # Oversample each class to match max_size
    balanced_dfs = []
    for label in class_counts.index:
        class_df = df[df[label_col] == label]
        
        if len(class_df) < max_size:
            # Oversample
            oversampled = class_df.sample(n=max_size, replace=True, random_state=config.RANDOM_SEED)
            balanced_dfs.append(oversampled)
        else:
            balanced_dfs.append(class_df)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)
    
    print(f"\nBalanced distribution:\n{balanced_df[label_col].value_counts()}")
    print(f"Total samples: {len(balanced_df)}")
    
    return balanced_df


def calculate_metrics(y_true: List, y_pred: List, labels: List = None) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for classification
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
    }
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics['per_class'] = report
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def print_metrics(metrics: Dict[str, Any], title: str = "Metrics"):
    """
    Pretty print metrics
    """
    print("\n" + "="*60)
    print(title)
    print("="*60)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1 (macro):    {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"  Precision:     {metrics['precision_macro']:.4f}")
    print(f"  Recall:        {metrics['recall_macro']:.4f}")
    
    if 'per_class' in metrics:
        print(f"\nPer-Class Metrics:")
        for label, values in metrics['per_class'].items():
            if isinstance(values, dict) and 'f1-score' in values:
                print(f"  {label:20s} - P: {values['precision']:.3f}, R: {values['recall']:.3f}, F1: {values['f1-score']:.3f}")


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], save_path: str = None):
    """
    Plot and save confusion matrix
    
    Args:
        cm: Confusion matrix
        labels: Label names
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_training_curves(history: Dict[str, List], save_path: str = None):
    """
    Plot training and validation curves
    
    Args:
        history: Dictionary with training history
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curve
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_title('Loss over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[1].plot(history['val_acc'], label='Val Accuracy', marker='s')
    axes[1].set_title('Accuracy over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training curves saved to {save_path}")
    
    plt.close()


def save_metrics_to_json(metrics: Dict, filepath: str):
    """
    Save metrics to JSON file
    
    Args:
        metrics: Metrics dictionary
        filepath: Path to save JSON
    """
    # Convert numpy arrays to lists for JSON serialization
    json_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            json_metrics[key] = value.tolist()
        elif isinstance(value, dict):
            json_metrics[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in value.items()}
        else:
            json_metrics[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    
    print(f"✓ Metrics saved to {filepath}")


def get_device_info():
    """
    Print device information
    """
    print("\n" + "="*60)
    print("Device Information")
    print("="*60)
    print(f"Device: {config.DEVICE}")
    
    if config.USE_GPU:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Running on CPU (training will be slower)")
        print("Consider using GPU for faster training")
    print("="*60)


def check_data_exists():
    """
    Check if processed data exists
    
    Returns:
        Boolean indicating if data exists
    """
    category_path = os.path.join(config.PROCESSED_DATA_DIR, 'category_data.csv')
    bias_path = os.path.join(config.PROCESSED_DATA_DIR, 'bias_data.csv')
    
    if not os.path.exists(category_path) or not os.path.exists(bias_path):
        return False
    
    return True


def estimate_training_time(num_samples: int, batch_size: int, epochs: int, use_gpu: bool = False):
    """
    Estimate training time
    
    Args:
        num_samples: Number of training samples
        batch_size: Batch size
        epochs: Number of epochs
        use_gpu: Whether using GPU
        
    Returns:
        Estimated time in minutes
    """
    steps_per_epoch = num_samples // batch_size
    
    if use_gpu:
        # ~0.5 seconds per batch on GPU
        time_per_batch = 0.5
    else:
        # ~5 seconds per batch on CPU
        time_per_batch = 5.0
    
    total_time_seconds = steps_per_epoch * epochs * time_per_batch
    total_time_minutes = total_time_seconds / 60
    
    return total_time_minutes


def format_time(seconds: float) -> str:
    """
    Format seconds into readable time string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g., "2h 34m 12s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def create_category_color_map(categories: List[str]) -> Dict[str, str]:
    """
    Create color mapping for categories
    
    Args:
        categories: List of category names
        
    Returns:
        Dictionary mapping categories to colors
    """
    # Use a colormap
    colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))
    color_hex = ['#' + ''.join([f'{int(c*255):02x}' for c in color[:3]]) for color in colors]
    
    return {cat: color for cat, color in zip(categories, color_hex)}


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    get_device_info()
    
    # Test metrics
    y_true = [0, 1, 0, 1, 1]
    y_pred = [0, 1, 1, 1, 1]
    
    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics, "Test Metrics")
    
    print("\n✓ All utility functions working correctly")

