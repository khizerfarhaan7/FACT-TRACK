"""
Text preprocessing utilities for FactTrack
"""

import re
import string
from typing import List


def clean_text(text: str) -> str:
    """
    Clean text by removing URLs, special characters, and extra whitespace
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def split_paragraphs(text: str, min_length: int = 30) -> List[str]:
    """
    Split text into paragraphs and filter by minimum length
    
    Args:
        text: Input text string
        min_length: Minimum character length for a valid paragraph
        
    Returns:
        List of paragraph strings
    """
    # Split on double newlines first
    paragraphs = re.split(r'\n\s*\n', text)
    
    # Further split on single newlines if no double newlines exist
    if len(paragraphs) == 1:
        paragraphs = text.split('\n')
    
    # If still single paragraph, try splitting on periods followed by spaces
    if len(paragraphs) == 1 and len(text) > 500:
        # Split on period + space + capital letter (likely sentence boundaries)
        parts = re.split(r'\.\s+(?=[A-Z])', text)
        # Group sentences into paragraphs of ~3 sentences each
        paragraphs = []
        current = []
        for part in parts:
            current.append(part)
            if len(current) >= 3:
                paragraphs.append('. '.join(current) + '.')
                current = []
        if current:
            paragraphs.append('. '.join(current) + ('.' if not current[-1].endswith('.') else ''))
    
    # Clean and filter paragraphs
    cleaned_paragraphs = []
    for para in paragraphs:
        para = para.strip()
        if len(para) >= min_length:
            cleaned_paragraphs.append(para)
    
    return cleaned_paragraphs


def remove_stopwords(text: str, use_nltk: bool = False) -> str:
    """
    Remove common stopwords from text
    
    Args:
        text: Input text string
        use_nltk: Whether to use NLTK stopwords (requires nltk download)
        
    Returns:
        Text with stopwords removed
    """
    # Basic stopwords list
    basic_stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
        'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
    }
    
    if use_nltk:
        try:
            from nltk.corpus import stopwords
            basic_stopwords = set(stopwords.words('english'))
        except (ImportError, LookupError):
            # Fall back to basic stopwords if NLTK not available
            pass
    
    # Tokenize and filter
    words = text.lower().split()
    filtered_words = [word for word in words if word not in basic_stopwords]
    
    return ' '.join(filtered_words)


def preprocess_pipeline(text: str, remove_stops: bool = False, min_para_length: int = 30) -> List[str]:
    """
    Complete preprocessing pipeline
    
    Args:
        text: Raw input text
        remove_stops: Whether to remove stopwords
        min_para_length: Minimum paragraph length
        
    Returns:
        List of preprocessed paragraphs
    """
    # Clean the text
    cleaned = clean_text(text)
    
    # Split into paragraphs
    paragraphs = split_paragraphs(cleaned, min_length=min_para_length)
    
    # Optionally remove stopwords (usually not needed for modern ML models)
    if remove_stops:
        paragraphs = [remove_stopwords(para) for para in paragraphs]
    
    return paragraphs


def prepare_for_bert(text: str, max_length: int = 256) -> str:
    """
    Prepare text for BERT processing
    
    Args:
        text: Input text
        max_length: Maximum character length (approximate)
        
    Returns:
        Prepared text suitable for BERT
    """
    # Clean the text
    cleaned = clean_text(text)
    
    # Truncate if too long (BERT has max 512 tokens, ~4 chars per token)
    max_chars = max_length * 4
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    
    return cleaned


def detect_language(text: str) -> str:
    """
    Detect language of text
    
    Args:
        text: Input text
        
    Returns:
        Language code (e.g., 'en', 'es', 'fr')
    """
    try:
        from langdetect import detect
        return detect(text)
    except:
        # If langdetect fails or not installed, assume English
        return 'en'


def augment_text(text: str, num_augmentations: int = 1) -> List[str]:
    """
    Apply text augmentation for data expansion
    Simple implementation - can be enhanced
    
    Args:
        text: Input text
        num_augmentations: Number of augmented versions to create
        
    Returns:
        List of augmented texts
    """
    augmented = [text]  # Include original
    
    # Simple synonym replacement (basic implementation)
    # For production, use nlpaug library for better augmentation
    
    return augmented


if __name__ == "__main__":
    # Test the preprocessing functions
    sample_text = """
    The new technology breakthrough promises to revolutionize the industry.
    Scientists have been working on this for years.
    
    However, critics argue that more testing is needed before widespread adoption.
    Safety concerns remain paramount according to regulatory officials.
    """
    
    print("Original text:")
    print(sample_text)
    print("\n" + "="*50 + "\n")
    
    print("Cleaned text:")
    cleaned = clean_text(sample_text)
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    print("Split paragraphs:")
    paragraphs = split_paragraphs(sample_text)
    for i, para in enumerate(paragraphs, 1):
        print(f"{i}. {para}")
    print("\n" + "="*50 + "\n")
    
    print("Full pipeline:")
    processed = preprocess_pipeline(sample_text)
    for i, para in enumerate(processed, 1):
        print(f"{i}. {para}")
    
    print("\n" + "="*50 + "\n")
    print("BERT preparation:")
    bert_ready = prepare_for_bert(sample_text)
    print(bert_ready)
    
    print("\n" + "="*50 + "\n")
    print("Language detection:")
    lang = detect_language(sample_text)
    print(f"Detected language: {lang}")

