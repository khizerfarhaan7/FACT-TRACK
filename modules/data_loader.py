"""
Dataset loaders for BERT training in FactTrack
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import DistilBertTokenizer
from typing import List, Dict, Tuple
import config
import numpy as np


class NewsDataset(Dataset):
    """
    PyTorch Dataset for news category classification
    """
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer: DistilBertTokenizer, max_length: int = None):
        """
        Initialize dataset
        
        Args:
            texts: List of article texts
            labels: List of category labels (as integers)
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length or config.MAX_LENGTH
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class BiasDataset(Dataset):
    """
    PyTorch Dataset for bias detection
    """
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer: DistilBertTokenizer, max_length: int = None):
        """
        Initialize dataset
        
        Args:
            texts: List of article texts
            labels: List of bias labels (0 or 1)
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length or config.MAX_LENGTH
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_category_data(data_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and split category dataset
    
    Args:
        data_path: Path to category CSV file
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if data_path is None:
        data_path = f"{config.PROCESSED_DATA_DIR}/category_data.csv"
    
    print(f"\nLoading category data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"✓ Loaded {len(df)} articles")
    print(f"  Categories: {df['category'].nunique()} unique")
    
    # Convert categories to indices
    df['label'] = df['category'].map(config.CATEGORY_TO_IDX)
    
    # Remove any categories not in our config
    df = df[df['label'].notna()]
    df['label'] = df['label'].astype(int)
    
    # Shuffle
    df = df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)
    
    # Split
    n = len(df)
    train_end = int(n * config.TRAIN_RATIO)
    val_end = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))
    
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    print(f"\nData split:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    return train_df, val_df, test_df


def load_bias_data(data_path: str = None, balance: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and split bias dataset
    
    Args:
        data_path: Path to bias CSV file
        balance: Whether to balance classes
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if data_path is None:
        data_path = f"{config.PROCESSED_DATA_DIR}/bias_data.csv"
    
    print(f"\nLoading bias data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"✓ Loaded {len(df)} examples")
    print(f"  Bias distribution:")
    print(df['bias_label'].value_counts())
    
    # Convert bias labels to indices
    df['label'] = df['bias_label'].map(config.BIAS_TO_IDX)
    df = df[df['label'].notna()]
    df['label'] = df['label'].astype(int)
    
    # Balance if requested
    if balance and config.BALANCE_BIAS_CLASSES:
        print("\nBalancing bias classes...")
        biased_df = df[df['label'] == 1]
        not_biased_df = df[df['label'] == 0]
        
        min_size = min(len(biased_df), len(not_biased_df))
        
        biased_df = biased_df.sample(n=min_size, random_state=config.RANDOM_SEED)
        not_biased_df = not_biased_df.sample(n=min_size, random_state=config.RANDOM_SEED)
        
        df = pd.concat([biased_df, not_biased_df], ignore_index=True)
        df = df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)
        
        print(f"✓ Balanced dataset: {len(df)} examples")
        print(f"  Bias distribution:")
        print(df['bias_label'].value_counts())
    
    # Split
    n = len(df)
    train_end = int(n * config.TRAIN_RATIO)
    val_end = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))
    
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    print(f"\nData split:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    return train_df, val_df, test_df


def create_category_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                                tokenizer: DistilBertTokenizer, batch_size: int = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for category data
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        tokenizer: BERT tokenizer
        batch_size: Batch size
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    batch_size = batch_size or config.BATCH_SIZE
    
    train_dataset = NewsDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer
    )
    
    val_dataset = NewsDataset(
        texts=val_df['text'].tolist(),
        labels=val_df['label'].tolist(),
        tokenizer=tokenizer
    )
    
    test_dataset = NewsDataset(
        texts=test_df['text'].tolist(),
        labels=test_df['label'].tolist(),
        tokenizer=tokenizer
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader


def create_bias_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                            tokenizer: DistilBertTokenizer, batch_size: int = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for bias data
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        tokenizer: BERT tokenizer
        batch_size: Batch size
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    batch_size = batch_size or config.BATCH_SIZE
    
    train_dataset = BiasDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer
    )
    
    val_dataset = BiasDataset(
        texts=val_df['text'].tolist(),
        labels=val_df['label'].tolist(),
        tokenizer=tokenizer
    )
    
    test_dataset = BiasDataset(
        texts=test_df['text'].tolist(),
        labels=test_df['label'].tolist(),
        tokenizer=tokenizer
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    from transformers import DistilBertTokenizer
    
    print("Testing data loaders...")
    
    # Check if data exists
    import os
    if not os.path.exists(f"{config.PROCESSED_DATA_DIR}/category_data.csv"):
        print("Error: Please run download_data.py first to prepare the data")
        exit(1)
    
    # Load data
    train_df, val_df, test_df = load_category_data()
    
    # Create tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(config.BERT_MODEL)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_category_dataloaders(
        train_df, val_df, test_df, tokenizer, batch_size=4
    )
    
    # Test a batch
    batch = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  Input IDs shape: {batch['input_ids'].shape}")
    print(f"  Attention mask shape: {batch['attention_mask'].shape}")
    print(f"  Labels shape: {batch['label'].shape}")
    
    print("\n✓ Data loaders working correctly")

