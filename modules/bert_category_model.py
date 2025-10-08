"""
BERT-based Category Classification Model for FactTrack
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import DistilBertModel, DistilBertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple
import os
import config
from modules.utils import calculate_metrics, print_metrics


class BERTCategoryClassifier(nn.Module):
    """
    BERT-based multi-class classifier for news categories
    """
    
    def __init__(self, num_classes: int = None, dropout: float = 0.3):
        """
        Initialize BERT category classifier
        
        Args:
            num_classes: Number of categories to classify
            dropout: Dropout rate
        """
        super(BERTCategoryClassifier, self).__init__()
        
        self.num_classes = num_classes or len(config.CATEGORIES)
        
        # Load pretrained DistilBERT
        self.bert = DistilBertModel.from_pretrained(config.BERT_MODEL)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_classes)
        
        # Tokenizer
        self.tokenizer = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Logits for each class
        """
        # Get BERT embeddings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout and classifier
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def train_model(self, train_loader, val_loader, epochs: int = None, learning_rate: float = None):
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            learning_rate: Learning rate
            
        Returns:
            Training history
        """
        epochs = epochs or config.EPOCHS
        learning_rate = learning_rate or config.LEARNING_RATE
        
        # Setup
        device = torch.device(config.DEVICE)
        self.to(device)
        
        # Optimizer
        optimizer = AdamW(self.parameters(), lr=learning_rate)
        
        # Learning rate scheduler with warmup
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.WARMUP_STEPS,
            num_training_steps=total_steps
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'='*60}")
            
            # Training phase
            self.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc="Training")
            for batch in train_pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                logits = self(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                # Track metrics
                train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            
            # Validation phase
            val_loss, val_acc = self.evaluate(val_loader, criterion, device)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch results
            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc + config.MIN_DELTA:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                self.save_checkpoint(os.path.join(config.CHECKPOINTS_DIR, 'best_category_model.pt'))
                print(f"  ✓ New best validation accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{config.PATIENCE})")
            
            if patience_counter >= config.PATIENCE:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"Best Validation Accuracy: {best_val_acc:.4f}")
        print(f"{'='*60}")
        
        return self.history
    
    def evaluate(self, data_loader, criterion=None, device=None, return_metrics=False):
        """
        Evaluate the model
        
        Args:
            data_loader: Data loader
            criterion: Loss function
            device: Device to use
            return_metrics: Whether to return detailed metrics
            
        Returns:
            Tuple of (loss, accuracy) or (loss, accuracy, f1, metrics)
        """
        if device is None:
            device = torch.device(config.DEVICE)
        
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                logits = self(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total
        
        if return_metrics:
            from sklearn.metrics import f1_score
            from modules.utils import calculate_metrics
            
            f1 = f1_score(all_labels, all_preds, average='weighted')
            metrics = calculate_metrics(all_labels, all_preds, labels=list(range(self.num_classes)))
            return avg_loss, accuracy, f1, metrics
        
        return avg_loss, accuracy
    
    def predict(self, texts: List[str], return_top_n: int = 3) -> List[Dict]:
        """
        Predict categories for texts
        
        Args:
            texts: List of text samples
            return_top_n: Number of top predictions to return
            
        Returns:
            List of prediction dictionaries
        """
        if self.tokenizer is None:
            self.tokenizer = DistilBertTokenizer.from_pretrained(config.BERT_MODEL)
        
        device = torch.device(config.DEVICE)
        self.to(device)
        self.eval()
        
        results = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                encoding = self.tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=config.MAX_LENGTH,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                # Get predictions
                logits = self(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
                # Get top N predictions
                top_indices = np.argsort(probs)[-return_top_n:][::-1]
                
                top_categories = [
                    {
                        'category': config.IDX_TO_CATEGORY[idx],
                        'confidence': float(probs[idx])
                    }
                    for idx in top_indices
                ]
                
                results.append({
                    'top_categories': top_categories,
                    'all_probabilities': {
                        config.IDX_TO_CATEGORY[i]: float(probs[i])
                        for i in range(len(probs))
                    }
                })
        
        return results
    
    def save_model(self, directory: str):
        """
        Save model and tokenizer
        
        Args:
            directory: Directory to save files
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save model
        model_path = os.path.join(directory, 'model.pt')
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_classes': self.num_classes,
            'history': self.history
        }, model_path)
        
        # Save tokenizer
        if self.tokenizer is None:
            self.tokenizer = DistilBertTokenizer.from_pretrained(config.BERT_MODEL)
        
        self.tokenizer.save_pretrained(directory)
        
        print(f"✓ Category model saved to {directory}/")
    
    def load_model(self, directory: str):
        """
        Load model and tokenizer
        
        Args:
            directory: Directory containing model files
        """
        model_path = os.path.join(directory, 'model.pt')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load model (weights_only=False for our trusted models - PyTorch 2.6+ compatibility)
        checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
        self.num_classes = checkpoint['num_classes']
        
        # Rebuild classifier if needed
        if self.classifier.out_features != self.num_classes:
            self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_classes)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        # Load tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(directory)
        
        print(f"✓ Category model loaded from {directory}/")
    
    def save_checkpoint(self, filepath: str):
        """
        Save training checkpoint
        
        Args:
            filepath: Path to save checkpoint
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_classes': self.num_classes,
            'history': self.history
        }, filepath)


if __name__ == "__main__":
    # Test the model
    print("Testing BERT Category Classifier...")
    
    model = BERTCategoryClassifier(num_classes=len(config.CATEGORIES))
    print(f"✓ Model initialized with {len(config.CATEGORIES)} categories")
    
    # Test forward pass
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, 30000, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))
    
    logits = model(input_ids, attention_mask)
    print(f"✓ Forward pass successful. Output shape: {logits.shape}")
    
    print("\n✓ BERT Category Classifier working correctly")

