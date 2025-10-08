"""
Test script to verify FactTrack installation and basic functionality
Run this after training models to ensure everything works correctly
"""

import os
import sys


def check_dependencies():
    """Check if all required packages are installed"""
    print("Checking dependencies...")
    required_packages = [
        'flask',
        'flask_cors',
        'sklearn',
        'pandas',
        'numpy',
        'nltk',
        'joblib'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed\n")
    return True


def check_data():
    """Check if training data exists"""
    print("Checking training data...")
    data_path = 'data/sample_articles.csv'
    
    if not os.path.exists(data_path):
        print(f"  ✗ {data_path} - NOT FOUND")
        print("\n❌ Training data missing")
        return False
    
    print(f"  ✓ {data_path}")
    
    # Check data content
    try:
        import pandas as pd
        df = pd.read_csv(data_path)
        print(f"  ✓ Contains {len(df)} articles")
        print(f"  ✓ Categories: {df['category'].nunique()}")
        print(f"  ✓ Bias labels: {df['bias_label'].nunique()}")
        print("✅ Training data valid\n")
        return True
    except Exception as e:
        print(f"  ✗ Error reading data: {e}")
        print("\n❌ Training data invalid")
        return False


def check_models():
    """Check if trained models exist"""
    print("Checking trained models...")
    model_files = [
        'models/category_model.pkl',
        'models/category_vectorizer.pkl',
        'models/bias_model.pkl',
        'models/bias_vectorizer.pkl'
    ]
    
    missing = []
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"  ✓ {model_file}")
        else:
            print(f"  ✗ {model_file} - NOT FOUND")
            missing.append(model_file)
    
    if missing:
        print("\n⚠️  Models not found. Run: python train.py")
        return False
    
    print("✅ All models present\n")
    return True


def test_preprocessing():
    """Test text preprocessing module"""
    print("Testing preprocessing module...")
    try:
        from modules.preprocess import clean_text, split_paragraphs, preprocess_pipeline
        
        sample_text = "This is a test. Check if preprocessing works properly."
        
        # Test clean_text
        cleaned = clean_text(sample_text)
        assert len(cleaned) > 0, "clean_text failed"
        print("  ✓ clean_text()")
        
        # Test split_paragraphs
        paragraphs = split_paragraphs(sample_text)
        assert isinstance(paragraphs, list), "split_paragraphs failed"
        print("  ✓ split_paragraphs()")
        
        # Test pipeline
        result = preprocess_pipeline(sample_text)
        assert isinstance(result, list), "preprocess_pipeline failed"
        print("  ✓ preprocess_pipeline()")
        
        print("✅ Preprocessing module works\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        print("\n❌ Preprocessing module failed")
        return False


def test_models():
    """Test model loading and prediction"""
    print("Testing model modules...")
    
    if not check_models():
        print("⚠️  Skipping model tests (models not trained)\n")
        return False
    
    try:
        from modules.category_model import CategoryClassifier
        from modules.bias_model import BiasDetector
        
        # Test category classifier
        cat_classifier = CategoryClassifier()
        cat_classifier.load_model('models')
        
        test_text = "The government announced new policies today."
        result = cat_classifier.predict_single(test_text)
        
        assert 'category' in result, "Category prediction failed"
        assert 'confidence' in result, "Confidence score missing"
        print(f"  ✓ CategoryClassifier (predicted: {result['category']})")
        
        # Test bias detector
        bias_detector = BiasDetector()
        bias_detector.load_model('models')
        
        result = bias_detector.predict_single(test_text)
        
        assert 'bias' in result, "Bias prediction failed"
        assert 'confidence' in result, "Confidence score missing"
        print(f"  ✓ BiasDetector (predicted: {result['bias']})")
        
        print("✅ Model modules work\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        print("\n❌ Model modules failed")
        return False


def test_flask_imports():
    """Test Flask app imports"""
    print("Testing Flask application...")
    try:
        import app
        print("  ✓ app.py imports successfully")
        print("  ✓ Flask server can be started")
        print("✅ Flask application ready\n")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        print("\n❌ Flask application failed")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("FactTrack Installation Test")
    print("="*60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Dependencies", check_dependencies()))
    results.append(("Training Data", check_data()))
    results.append(("Trained Models", check_models()))
    results.append(("Preprocessing", test_preprocessing()))
    results.append(("Models", test_models()))
    results.append(("Flask App", test_flask_imports()))
    
    # Summary
    print("="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20s} {status}")
    
    print("="*60)
    print(f"Total: {passed}/{total} tests passed")
    print("="*60 + "\n")
    
    if passed == total:
        print("🎉 All tests passed! FactTrack is ready to use.")
        print("\nNext steps:")
        print("  1. Run: python app.py")
        print("  2. Open: http://localhost:5000")
        print("  3. Start analyzing news articles!")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Train models: python train.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())

