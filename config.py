"""
Configuration file for FactTrack Production System
"""

import torch
import os

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# BERT Model Selection
BERT_MODEL = "distilbert-base-uncased"  # Faster and lighter than full BERT
# Alternative: "bert-base-uncased" for better accuracy but slower

# Training Hyperparameters
MAX_LENGTH = 256  # Maximum sequence length for BERT
BATCH_SIZE = 16   # Batch size for training
LEARNING_RATE = 2e-5  # Learning rate for fine-tuning
EPOCHS = 3  # Number of training epochs
WARMUP_STEPS = 500  # Learning rate warmup steps

# Early Stopping
PATIENCE = 2  # Stop if no improvement for N epochs
MIN_DELTA = 0.001  # Minimum improvement to count as progress

# ============================================================================
# CATEGORIES (20+ categories for real-world news)
# ============================================================================

CATEGORIES = [
    'politics',
    'world_news',
    'business',
    'technology',
    'sports',
    'entertainment',
    'health',
    'science',
    'education',
    'environment',
    'crime',
    'economy',
    'finance',
    'travel',
    'food',
    'lifestyle',
    'real_estate',
    'automotive',
    'opinion',
    'local_news'
]

# Category to index mapping
CATEGORY_TO_IDX = {cat: idx for idx, cat in enumerate(CATEGORIES)}
IDX_TO_CATEGORY = {idx: cat for idx, cat in enumerate(CATEGORIES)}

# ============================================================================
# BIAS LABELS
# ============================================================================

BIAS_LABELS = ['not_biased', 'biased']
BIAS_TO_IDX = {label: idx for idx, label in enumerate(BIAS_LABELS)}
IDX_TO_BIAS = {idx: label for idx, label in enumerate(BIAS_LABELS)}

# ============================================================================
# PATHS
# ============================================================================

# Data directories
DATA_DIR = 'data'
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')

# Model directories
MODELS_DIR = 'models'
CATEGORY_MODEL_DIR = os.path.join(MODELS_DIR, 'category_bert')
BIAS_MODEL_DIR = os.path.join(MODELS_DIR, 'bias_bert')
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, 'checkpoints')

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR, 
                  MODELS_DIR, CATEGORY_MODEL_DIR, BIAS_MODEL_DIR, CHECKPOINTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

# Auto-detect CUDA availability
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_GPU = torch.cuda.is_available()

# Print device info
print(f"Using device: {DEVICE}")
if USE_GPU:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Dataset sizes
TARGET_CATEGORY_SAMPLES = 3000  # Target number of samples for category training
TARGET_BIAS_SAMPLES = 2000  # Target number of samples for bias training
MIN_TEXT_LENGTH = 50  # Minimum character length for valid articles
MAX_TEXT_LENGTH = 5000  # Maximum character length

# Train/Val/Test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Data balancing
BALANCE_BIAS_CLASSES = True  # Ensure 50/50 split for bias detection
OVERSAMPLE_MINORITY = True  # Oversample minority class if needed

# ============================================================================
# BIAS DETECTION PARAMETERS
# ============================================================================

# Class weights for imbalanced data
BIAS_CLASS_WEIGHTS = torch.tensor([1.0, 1.0])  # Will be adjusted based on data

# Bias threshold
BIAS_THRESHOLD = 0.5  # Probability threshold for classifying as biased

# Focal loss parameters (for handling class imbalance)
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0

# ============================================================================
# API CONFIGURATION
# ============================================================================

# Flask settings
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True

# API limits
MAX_PARAGRAPHS_PER_REQUEST = 50
MAX_TEXT_LENGTH_API = 50000

# Response configuration
RETURN_TOP_N_CATEGORIES = 3  # Return top N category predictions

# ============================================================================
# DATASET URLS (for download_data.py)
# ============================================================================

# Hugging Face datasets
AGNEWS_DATASET = "ag_news"
NEWS_CATEGORY_DATASET = "SetFit/20_newsgroups"

# Note: Some datasets require manual download or kaggle API
# Hyperpartisan News: https://pan.webis.de/semeval19/semeval19-web/
# All The News: https://www.kaggle.com/snapcrack/all-the-news

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Gradient accumulation (for larger effective batch size)
GRADIENT_ACCUMULATION_STEPS = 2

# Mixed precision training (faster on modern GPUs)
USE_AMP = True if USE_GPU else False

# Logging
LOG_EVERY_N_STEPS = 50
SAVE_CHECKPOINT_EVERY_N_EPOCHS = 1

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

# Metrics to track
TRACK_METRICS = ['accuracy', 'f1', 'precision', 'recall']

# Confusion matrix
SAVE_CONFUSION_MATRIX = True

# ============================================================================
# PREPROCESSING CONFIGURATION
# ============================================================================

# Text cleaning
REMOVE_URLS = True
REMOVE_EMAILS = True
REMOVE_SPECIAL_CHARS = False  # Keep for BERT
LOWERCASE = False  # BERT handles casing

# Language detection
DETECT_LANGUAGE = True
TARGET_LANGUAGE = 'en'

# ============================================================================
# CACHE CONFIGURATION
# ============================================================================

# Hugging Face cache
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_HOME'] = CACHE_DIR

# Disable tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ============================================================================
# RANDOM SEED (for reproducibility)
# ============================================================================

RANDOM_SEED = 42

# Set seeds
torch.manual_seed(RANDOM_SEED)
if USE_GPU:
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

import numpy as np
import random

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ============================================================================
# VERSION INFO
# ============================================================================

VERSION = "2.0.0"
MODEL_VERSION = "BERT-Production"

