"""
BERT-based Bias Detection Model for FactTrack

This model includes critical fixes for bias detection:
1. Weighted loss function for class imbalance
2. Focal loss option for hard examples
3. F1 score monitoring (not just accuracy)
4. Proper evaluation with precision/recall
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import DistilBertModel, DistilBertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple
import os
import config
from modules.utils import calculate_metrics, print_metrics


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class BERTBiasDetector(nn.Module):
    """
    BERT-based binary classifier for bias detection
    WITH FIXES for class imbalance and proper evaluation
    """
    
    def __init__(self, dropout: float = 0.3, use_focal_loss: bool = False):
        """
        Initialize BERT bias detector
        
        Args:
            dropout: Dropout rate
            use_focal_loss: Whether to use focal loss instead of cross entropy
        """
        super(BERTBiasDetector, self).__init__()
        
        # Load pretrained DistilBERT
        self.bert = DistilBertModel.from_pretrained(config.BERT_MODEL)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)  # Binary classification
        
        # Tokenizer
        self.tokenizer = None
        
        # Use focal loss
        self.use_focal_loss = use_focal_loss
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_f1': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Logits for binary classification
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
        Train the bias detection model
        
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
        
        # Loss function with class weights (CRITICAL FIX)
        if self.use_focal_loss:
            criterion = FocalLoss(alpha=config.FOCAL_LOSS_ALPHA, gamma=config.FOCAL_LOSS_GAMMA)
            print("Using Focal Loss for training")
        else:
            # Calculate class weights from training data
            class_counts = self._calculate_class_weights(train_loader, device)
            weights = torch.tensor([1.0 / count for count in class_counts]).to(device)
            weights = weights / weights.sum() * len(weights)  # Normalize
            criterion = nn.CrossEntropyLoss(weight=weights)
            print(f"Using Weighted Cross Entropy Loss with weights: {weights.cpu().numpy()}")
        
        # Training loop
        best_val_f1 = 0.0  # Track F1 instead of accuracy for imbalanced data
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
            all_train_preds = []
            all_train_labels = []
            
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
                
                # Store for F1 calculation
                all_train_preds.extend(predicted.cpu().numpy())
                all_train_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            
            # Calculate F1 score
            from sklearn.metrics import f1_score
            train_f1 = f1_score(all_train_labels, all_train_preds, average='binary')
            
            # Validation phase
            val_loss, val_acc, val_f1, val_metrics = self.evaluate(val_loader, criterion, device, return_metrics=True)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            
            # Print epoch results
            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            
            if val_metrics:
                print(f"  Val Precision: {val_metrics['precision_macro']:.4f}")
                print(f"  Val Recall:    {val_metrics['recall_macro']:.4f}")
            
            # Early stopping based on F1 score (better for imbalanced data)
            if val_f1 > best_val_f1 + config.MIN_DELTA:
                best_val_f1 = val_f1
                patience_counter = 0
                # Save best model
                self.save_checkpoint(os.path.join(config.CHECKPOINTS_DIR, 'best_bias_model.pt'))
                print(f"  ✓ New best validation F1: {val_f1:.4f}")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{config.PATIENCE})")
            
            if patience_counter >= config.PATIENCE:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"Best Validation F1 Score: {best_val_f1:.4f}")
        print(f"{'='*60}")
        
        return self.history
    
    def _calculate_class_weights(self, data_loader, device):
        """
        Calculate class weights from data loader
        """
        class_counts = [0, 0]
        for batch in data_loader:
            labels = batch['label']
            for label in labels:
                class_counts[label.item()] += 1
        return class_counts
    
    def evaluate(self, data_loader, criterion=None, device=None, return_metrics=False):
        """
        Evaluate the model
        
        Args:
            data_loader: Data loader
            criterion: Loss function
            device: Device to use
            return_metrics: Whether to return detailed metrics
            
        Returns:
            Tuple of (loss, accuracy, f1_score) or (loss, accuracy, f1_score, metrics)
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
        
        # Calculate F1 score
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average='binary')
        
        if return_metrics:
            metrics = calculate_metrics(all_labels, all_preds, labels=[0, 1])
            return avg_loss, accuracy, f1, metrics
        
        return avg_loss, accuracy, f1
    
    def predict(self, texts: List[str]) -> List[Dict]:
        """
        Predict bias for texts
        
        Args:
            texts: List of text samples
            
        Returns:
            List of prediction dictionaries with bias label, probability, and confidence
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
                
                # Get prediction
                predicted_class = np.argmax(probs)
                bias_label = config.IDX_TO_BIAS[predicted_class]
                confidence = float(probs[predicted_class])
                bias_probability = float(probs[1])  # Probability of being biased
                
                # Determine confidence level
                if confidence > 0.8:
                    confidence_level = "high"
                elif confidence > 0.6:
                    confidence_level = "medium"
                else:
                    confidence_level = "low"
                
                results.append({
                    'bias': bias_label,
                    'probability': bias_probability,
                    'confidence': confidence,
                    'confidence_level': confidence_level,
                    'not_biased_probability': float(probs[0])
                })
        
        return results
    
    def get_bias_indicators(self, text: str) -> List[str]:
        """
        Get potential bias indicator words/phrases
        This is a simple implementation - could be enhanced with attention weights
        
        Args:
            text: Input text
            
        Returns:
            List of potential bias indicator words
        """
        # Simple keyword-based bias indicators
        bias_keywords = [
            'corrupt', 'evil', 'terrible', 'awful', 'disaster', 'catastrophe',
            'shocking', 'unbelievable', 'insane', 'crazy', 'radical', 'extremist',
            'incompetent', 'failed', 'devastating', 'heroic', 'courageous',
            'villainous', 'barbaric', 'ruthless', 'beloved', 'stunning'
        ]
        
        text_lower = text.lower()
        found_indicators = []
        
        for keyword in bias_keywords:
            if keyword in text_lower:
                found_indicators.append(keyword)
        
        return found_indicators
    
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
            'use_focal_loss': self.use_focal_loss,
            'history': self.history
        }, model_path)
        
        # Save tokenizer
        if self.tokenizer is None:
            self.tokenizer = DistilBertTokenizer.from_pretrained(config.BERT_MODEL)
        
        self.tokenizer.save_pretrained(directory)
        
        print(f"✓ Bias model saved to {directory}/")
    
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
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if 'use_focal_loss' in checkpoint:
            self.use_focal_loss = checkpoint['use_focal_loss']
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        # Load tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(directory)
        
        print(f"✓ Bias model loaded from {directory}/")
    
    def save_checkpoint(self, filepath: str):
        """
        Save training checkpoint
        
        Args:
            filepath: Path to save checkpoint
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'use_focal_loss': self.use_focal_loss,
            'history': self.history
        }, filepath)


if __name__ == "__main__":
    # Test the model
    print("Testing BERT Bias Detector...")
    
    model = BERTBiasDetector()
    print(f"✓ Model initialized")
    
    # Test forward pass
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, 30000, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))
    
    logits = model(input_ids, attention_mask)
    print(f"✓ Forward pass successful. Output shape: {logits.shape}")
    
    print("\n✓ BERT Bias Detector working correctly")

