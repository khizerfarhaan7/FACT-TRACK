"""
FactTrack Flask Backend Server - Production BERT Version

API endpoints for news classification and bias detection using BERT models
"""

import os
import time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import config
from modules.bert_category_model import BERTCategoryClassifier
from modules.bert_bias_model import BERTBiasDetector
from modules.preprocess import preprocess_pipeline

# Initialize Flask app
app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)  # Enable CORS for all routes

# Global model instances
category_model = None
bias_model = None
models_loaded = False
model_info = {}


def load_models():
    """
    Load pre-trained BERT models on application startup
    """
    global category_model, bias_model, models_loaded, model_info
    
    print("\n" + "="*60)
    print("Loading FactTrack BERT Models...")
    print("="*60)
    
    try:
        # Check if model directories exist
        if not os.path.exists(config.CATEGORY_MODEL_DIR) or not os.path.exists(config.BIAS_MODEL_DIR):
            print("✗ Model directories not found!")
            print(f"  Category model: {config.CATEGORY_MODEL_DIR}")
            print(f"  Bias model: {config.BIAS_MODEL_DIR}")
            print("\nPlease train the models first:")
            print("  1. python download_data.py")
            print("  2. python train.py")
            return False
        
        # Load category classifier
        print("\n1. Loading category classifier...")
        category_model = BERTCategoryClassifier(num_classes=len(config.CATEGORIES))
        category_model.load_model(config.CATEGORY_MODEL_DIR)
        category_model.eval()  # Set to evaluation mode
        print("   ✓ Category classifier loaded")
        
        # Load bias detector
        print("\n2. Loading bias detector...")
        bias_model = BERTBiasDetector()
        bias_model.load_model(config.BIAS_MODEL_DIR)
        bias_model.eval()  # Set to evaluation mode
        print("   ✓ Bias detector loaded")
        
        # Store model info
        model_info = {
            'model_type': 'DistilBERT',
            'bert_model': config.BERT_MODEL,
            'num_categories': len(config.CATEGORIES),
            'categories': config.CATEGORIES,
            'device': config.DEVICE,
            'version': config.VERSION
        }
        
        models_loaded = True
        print("\n" + "="*60)
        print("✓ All models loaded successfully!")
        print(f"  Device: {config.DEVICE}")
        print(f"  Categories: {len(config.CATEGORIES)}")
        print(f"  Version: {config.VERSION}")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error loading models: {str(e)}")
        print("\nPlease ensure models are trained:")
        print("  python train.py")
        import traceback
        traceback.print_exc()
        return False


# Load models on startup
load_models()


@app.route('/')
def index():
    """
    Serve the main HTML page
    """
    return send_from_directory('frontend', 'index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify models are loaded
    """
    return jsonify({
        'status': 'healthy' if models_loaded else 'unhealthy',
        'models_loaded': models_loaded,
        'model_info': model_info if models_loaded else {},
        'message': 'Models loaded and ready' if models_loaded else 'Models not loaded. Please train models first.'
    })


@app.route('/api/categories', methods=['GET'])
def get_categories():
    """
    Get list of all supported categories
    """
    return jsonify({
        'success': True,
        'categories': config.CATEGORIES,
        'total': len(config.CATEGORIES)
    })


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """
    Get detailed model information
    """
    if not models_loaded:
        return jsonify({
            'success': False,
            'error': 'Models not loaded'
        }), 503
    
    # Load metrics if available
    metrics = {}
    try:
        import json
        cat_metrics_path = os.path.join(config.MODELS_DIR, 'category_metrics.json')
        bias_metrics_path = os.path.join(config.MODELS_DIR, 'bias_metrics.json')
        
        if os.path.exists(cat_metrics_path):
            with open(cat_metrics_path, 'r') as f:
                metrics['category'] = json.load(f)
        
        if os.path.exists(bias_metrics_path):
            with open(bias_metrics_path, 'r') as f:
                metrics['bias'] = json.load(f)
    except:
        pass
    
    return jsonify({
        'success': True,
        'model_info': model_info,
        'metrics': metrics
    })


@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    """
    Analyze news article for category and bias using BERT models
    
    Expected JSON input:
    {
        "text": "Article text to analyze..."
    }
    
    Returns JSON with top-3 categories and bias detection
    """
    start_time = time.time()
    
    try:
        # Check if models are loaded
        if not models_loaded:
            return jsonify({
                'success': False,
                'error': 'Models not loaded. Please train models first: python train.py'
            }), 503
        
        # Get input data
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: text'
            }), 400
        
        input_text = data['text']
        
        # Validate input
        if not input_text or not input_text.strip():
            return jsonify({
                'success': False,
                'error': 'Text field cannot be empty'
            }), 400
        
        if len(input_text) < 30:
            return jsonify({
                'success': False,
                'error': 'Text is too short. Please provide at least 30 characters.'
            }), 400
        
        if len(input_text) > config.MAX_TEXT_LENGTH_API:
            return jsonify({
                'success': False,
                'error': f'Text is too long. Maximum length is {config.MAX_TEXT_LENGTH_API} characters.'
            }), 400
        
        # Preprocess text into paragraphs
        paragraphs = preprocess_pipeline(input_text, remove_stops=False, min_para_length=30)
        
        if not paragraphs:
            return jsonify({
                'success': False,
                'error': 'No valid paragraphs found. Text may be too short or improperly formatted.'
            }), 400
        
        if len(paragraphs) > config.MAX_PARAGRAPHS_PER_REQUEST:
            paragraphs = paragraphs[:config.MAX_PARAGRAPHS_PER_REQUEST]
        
        # Analyze with BERT models
        results = []
        
        # Get predictions from both models
        category_predictions = category_model.predict(paragraphs, return_top_n=config.RETURN_TOP_N_CATEGORIES)
        bias_predictions = bias_model.predict(paragraphs)
        
        # Combine results
        for para, cat_pred, bias_pred in zip(paragraphs, category_predictions, bias_predictions):
            # Get bias indicators
            bias_indicators = bias_model.get_bias_indicators(para)
            
            results.append({
                'paragraph': para,
                'top_categories': cat_pred['top_categories'],
                'bias': {
                    'label': bias_pred['bias'],
                    'probability': round(bias_pred['probability'], 4),
                    'confidence': round(bias_pred['confidence'], 4),
                    'confidence_level': bias_pred['confidence_level'],
                    'not_biased_probability': round(bias_pred['not_biased_probability'], 4),
                    'indicators': bias_indicators[:5] if bias_indicators else []
                }
            })
        
        processing_time = int((time.time() - start_time) * 1000)  # Convert to milliseconds
        
        return jsonify({
            'success': True,
            'model': config.BERT_MODEL,
            'version': config.VERSION,
            'results': results,
            'total_paragraphs': len(results),
            'processing_time_ms': processing_time
        })
    
    except Exception as e:
        print(f"Error in analyze_text: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500


@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """
    Test endpoint for debugging
    """
    return jsonify({
        'message': 'FactTrack API is running',
        'version': config.VERSION,
        'models_loaded': models_loaded,
        'device': config.DEVICE
    })


@app.errorhandler(404)
def not_found(e):
    """
    Handle 404 errors
    """
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(e):
    """
    Handle 500 errors
    """
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    if not models_loaded:
        print("\n" + "="*60)
        print("⚠️ WARNING: Models are not loaded!")
        print("="*60)
        print("\nPlease follow these steps:")
        print("  1. python download_data.py  (download datasets)")
        print("  2. python train.py          (train BERT models)")
        print("  3. python app.py            (start server)")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("FactTrack Production Server Starting...")
        print("="*60)
        print(f"\nVersion: {config.VERSION}")
        print(f"Model: {config.BERT_MODEL}")
        print(f"Device: {config.DEVICE}")
        print(f"Categories: {len(config.CATEGORIES)}")
        print("\n✓ Models loaded successfully")
        print(f"\n🌐 Access the application at: http://localhost:{config.FLASK_PORT}")
        print("="*60 + "\n")
    
    # Run the Flask application
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)
