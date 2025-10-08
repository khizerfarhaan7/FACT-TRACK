"""
BERT Training Script for FactTrack Production System

This script trains both the category classifier and bias detector
using real datasets and BERT models.
"""

import os
import torch
import time
from transformers import DistilBertTokenizer
import config
from modules.bert_category_model import BERTCategoryClassifier
from modules.bert_bias_model import BERTBiasDetector
from modules.data_loader import (
    load_category_data, load_bias_data,
    create_category_dataloaders, create_bias_dataloaders
)
from modules.utils import (
    get_device_info, check_data_exists, print_metrics,
    plot_training_curves, plot_confusion_matrix,
    save_metrics_to_json, estimate_training_time, format_time
)
from sklearn.metrics import confusion_matrix
import numpy as np


def train_category_model():
    """
    Train the BERT category classification model
    """
    print("\n" + "="*80)
    print("TRAINING CATEGORY CLASSIFIER (BERT)")
    print("="*80)
    
    # Load data
    print("\nLoading category data...")
    train_df, val_df, test_df = load_category_data()
    
    # Initialize tokenizer
    print("\nInitializing BERT tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(config.BERT_MODEL)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_category_dataloaders(
        train_df, val_df, test_df, tokenizer
    )
    
    print(f"\nDataloaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Estimate training time
    est_time = estimate_training_time(
        num_samples=len(train_df),
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        use_gpu=config.USE_GPU
    )
    print(f"\nEstimated training time: ~{int(est_time)} minutes")
    
    # Initialize model
    print(f"\nInitializing BERT model ({config.BERT_MODEL})...")
    model = BERTCategoryClassifier(num_classes=len(config.CATEGORIES))
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print(f"\nStarting training for {config.EPOCHS} epochs...")
    print("This may take a while. Progress will be shown below.\n")
    
    start_time = time.time()
    history = model.train_model(train_loader, val_loader)
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {format_time(training_time)}")
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80)
    
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, test_f1, test_metrics = model.evaluate(
        test_loader, criterion, return_metrics=True
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    
    # Print detailed metrics
    print_metrics(test_metrics, "Category Classification - Test Set")
    
    # Plot confusion matrix
    if config.SAVE_CONFUSION_MATRIX:
        cm = np.array(test_metrics['confusion_matrix'])
        plot_confusion_matrix(
            cm,
            config.CATEGORIES,
            save_path=os.path.join(config.MODELS_DIR, 'category_confusion_matrix.png')
        )
    
    # Plot training curves
    plot_training_curves(
        history,
        save_path=os.path.join(config.MODELS_DIR, 'category_training_curves.png')
    )
    
    # Save metrics
    test_metrics['training_time_seconds'] = training_time
    test_metrics['epochs_trained'] = len(history['train_loss'])
    save_metrics_to_json(
        test_metrics,
        os.path.join(config.MODELS_DIR, 'category_metrics.json')
    )
    
    # Save model
    print("\n" + "="*80)
    print("SAVING CATEGORY MODEL")
    print("="*80)
    model.save_model(config.CATEGORY_MODEL_DIR)
    
    return model, test_metrics


def train_bias_model():
    """
    Train the BERT bias detection model with fixes
    """
    print("\n" + "="*80)
    print("TRAINING BIAS DETECTOR (BERT)")
    print("="*80)
    print("\nThis model includes fixes for class imbalance:")
    print("  ✓ Balanced training data")
    print("  ✓ Weighted loss function")
    print("  ✓ F1 score monitoring")
    
    # Load data
    print("\nLoading bias data...")
    train_df, val_df, test_df = load_bias_data(balance=True)
    
    # Initialize tokenizer
    print("\nInitializing BERT tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(config.BERT_MODEL)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_bias_dataloaders(
        train_df, val_df, test_df, tokenizer
    )
    
    print(f"\nDataloaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Estimate training time
    est_time = estimate_training_time(
        num_samples=len(train_df),
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        use_gpu=config.USE_GPU
    )
    print(f"\nEstimated training time: ~{int(est_time)} minutes")
    
    # Initialize model
    print(f"\nInitializing BERT model ({config.BERT_MODEL})...")
    model = BERTBiasDetector(use_focal_loss=False)  # Use weighted CE loss
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print(f"\nStarting training for {config.EPOCHS} epochs...")
    print("This may take a while. Progress will be shown below.\n")
    
    start_time = time.time()
    history = model.train_model(train_loader, val_loader)
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {format_time(training_time)}")
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80)
    
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, test_f1, test_metrics = model.evaluate(
        test_loader, criterion, return_metrics=True
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1 Score: {test_f1:.4f} (CRITICAL METRIC FOR BIAS)")
    
    # Print detailed metrics
    print_metrics(test_metrics, "Bias Detection - Test Set")
    
    # Print per-class metrics explicitly
    print("\nPer-Class Performance:")
    for label in ['not_biased', 'biased']:
        if label in test_metrics['per_class']:
            metrics = test_metrics['per_class'][label]
            print(f"  {label:15s} - P: {metrics['precision']:.3f}, R: {metrics['recall']:.3f}, F1: {metrics['f1-score']:.3f}, Support: {metrics['support']}")
    
    # Plot confusion matrix
    if config.SAVE_CONFUSION_MATRIX:
        cm = np.array(test_metrics['confusion_matrix'])
        plot_confusion_matrix(
            cm,
            ['not_biased', 'biased'],
            save_path=os.path.join(config.MODELS_DIR, 'bias_confusion_matrix.png')
        )
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print("                  Predicted")
        print("               Not Biased  Biased")
        print(f"Actual Not Biased    {cm[0][0]:3d}      {cm[0][1]:3d}")
        print(f"       Biased        {cm[1][0]:3d}      {cm[1][1]:3d}")
    
    # Plot training curves
    plot_training_curves(
        history,
        save_path=os.path.join(config.MODELS_DIR, 'bias_training_curves.png')
    )
    
    # Save metrics
    test_metrics['training_time_seconds'] = training_time
    test_metrics['epochs_trained'] = len(history['train_loss'])
    save_metrics_to_json(
        test_metrics,
        os.path.join(config.MODELS_DIR, 'bias_metrics.json')
    )
    
    # Save model
    print("\n" + "="*80)
    print("SAVING BIAS MODEL")
    print("="*80)
    model.save_model(config.BIAS_MODEL_DIR)
    
    return model, test_metrics


def main():
    """
    Main training pipeline
    """
    print("="*80)
    print("FactTrack Production Training - BERT Models")
    print(f"Version: {config.VERSION}")
    print("="*80)
    
    # Print device info
    get_device_info()
    
    # Check if data exists
    if not check_data_exists():
        print("\n" + "="*80)
        print("ERROR: Training data not found!")
        print("="*80)
        print("\nPlease run the following command first:")
        print("  python download_data.py")
        print("\nThis will download and prepare the training datasets.")
        return 1
    
    print("\n✓ Training data found")
    
    # Ask for confirmation
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"BERT Model: {config.BERT_MODEL}")
    print(f"Device: {config.DEVICE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Max Length: {config.MAX_LENGTH}")
    print(f"\nCategories: {len(config.CATEGORIES)}")
    print(f"Bias Classes: {len(config.BIAS_LABELS)}")
    
    if not config.USE_GPU:
        print("\n⚠️ WARNING: Training on CPU will be slow (2-4 hours)")
        print("   Consider using GPU for faster training (30-60 minutes)")
    
    print("\n" + "="*80)
    
    # Track total time
    total_start = time.time()
    
    try:
        # Train category model
        cat_model, cat_metrics = train_category_model()
        
        # Train bias model
        bias_model, bias_metrics = train_bias_model()
        
        # Final summary
        total_time = time.time() - total_start
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        
        print(f"\nTotal training time: {format_time(total_time)}")
        
        print("\nCategory Classifier:")
        print(f"  Test Accuracy: {cat_metrics['accuracy']:.4f}")
        print(f"  Test F1 Score: {cat_metrics['f1_weighted']:.4f}")
        print(f"  Model saved to: {config.CATEGORY_MODEL_DIR}/")
        
        print("\nBias Detector:")
        print(f"  Test Accuracy: {bias_metrics['accuracy']:.4f}")
        print(f"  Test F1 Score: {bias_metrics['f1_macro']:.4f}")
        print(f"  Model saved to: {config.BIAS_MODEL_DIR}/")
        
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1. Run the Flask application:")
        print("   python app.py")
        print("\n2. Open your browser:")
        print("   http://localhost:5000")
        print("\n3. Start analyzing news articles!")
        
        print("\n" + "="*80)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
