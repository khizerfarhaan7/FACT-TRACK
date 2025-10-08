# 📁 FACT-TRACK Project Structure

Clean, organized structure for the FACT-TRACK repository.

## 🗂️ Directory Overview

```
FACT-TRACK/
├── 📄 Core Application Files
│   ├── app.py                          # Main Flask application
│   ├── config.py                       # Configuration settings
│   └── requirements.txt                # Python dependencies
│   └── requirements-gpu.txt            # GPU-optimized dependencies
│
├── 🤖 Machine Learning Components
│   ├── modules/                        # Core ML modules
│   │   ├── __init__.py
│   │   ├── bert_bias_model.py          # BERT bias detection model
│   │   ├── bert_category_model.py      # BERT category classification model
│   │   ├── data_loader.py              # Data loading utilities
│   │   ├── preprocess.py               # Data preprocessing
│   │   └── utils.py                    # Utility functions
│   │
│   ├── models/                         # Trained model files
│   │   ├── bias_bert/                  # Bias detection model
│   │   │   ├── model.pt
│   │   │   ├── tokenizer_config.json
│   │   │   ├── special_tokens_map.json
│   │   │   └── vocab.txt
│   │   ├── category_bert/              # Category classification model
│   │   │   ├── model.pt
│   │   │   ├── tokenizer_config.json
│   │   │   ├── special_tokens_map.json
│   │   │   └── vocab.txt
│   │   ├── checkpoints/                # Model checkpoints
│   │   │   ├── best_bias_model.pt
│   │   │   └── best_category_model.pt
│   │   ├── *_confusion_matrix.png      # Model evaluation plots
│   │   ├── *_training_curves.png       # Training progress plots
│   │   └── *_metrics.json              # Model performance metrics
│   │
│   └── scripts/                        # Training and utility scripts
│       ├── download_data.py            # Download training datasets
│       ├── train.py                    # Train BERT models
│       └── test_installation.py        # Test system setup
│
├── 🌐 Frontend
│   └── frontend/                       # Web interface
│       ├── index.html                  # Main HTML page
│       ├── style.css                   # Styling and animations
│       └── script.js                   # Frontend JavaScript
│
├── 📊 Data
│   └── data/
│       └── sample_articles.csv         # Sample news articles
│
├── 📚 Documentation
│   ├── docs/
│   │   └── INSTALLATION.md             # Installation guide
│   ├── README.md                       # Main project documentation
│   ├── PROJECT_STRUCTURE.md            # This file
│   ├── SECURITY.md                     # Security policy
│   ├── SECURITY_IMPLEMENTATION_GUIDE.md # Security setup guide
│   └── CONTRIBUTING.md                 # Contribution guidelines
│
├── 🔒 Security & Configuration
│   ├── .github/                        # GitHub configuration
│   │   ├── CODEOWNERS                  # Code ownership rules
│   │   └── workflows/                  # GitHub Actions workflows
│   │       ├── security-scan.yml       # Daily security scans
│   │       ├── dependency-update.yml   # Weekly dependency checks
│   │       └── pr-validation.yml       # Pull request validation
│   ├── .gitignore                      # Git ignore rules
│   └── Procfile                        # Deployment configuration
│
└── 📄 Legal & Metadata
    └── LICENSE                          # MIT License
```

## 🎯 File Categories

### **Core Application**
- `app.py` - Main Flask server and API endpoints
- `config.py` - Configuration settings (categories, model parameters)
- `requirements*.txt` - Python package dependencies

### **Machine Learning**
- `modules/` - BERT model implementations and utilities
- `models/` - Trained model files and evaluation results
- `scripts/` - Training, data download, and testing scripts

### **Frontend**
- `frontend/` - Complete web interface with modern UI

### **Data**
- `data/` - Sample articles and training datasets

### **Documentation**
- `README.md` - Main project documentation
- `docs/` - Detailed guides and documentation
- `SECURITY*.md` - Security policies and implementation guides
- `CONTRIBUTING.md` - Contribution guidelines

### **Security & DevOps**
- `.github/` - GitHub Actions workflows and code ownership
- `.gitignore` - Git ignore rules for clean repository
- `Procfile` - Deployment configuration

## 🚀 Quick Navigation

### **Start Here**
- `README.md` - Project overview and quick start
- `docs/INSTALLATION.md` - Detailed installation guide

### **Development**
- `app.py` - Main application entry point
- `config.py` - Configuration settings
- `modules/` - Core ML implementation

### **Training**
- `scripts/download_data.py` - Download training data
- `scripts/train.py` - Train BERT models
- `models/` - Trained model outputs

### **Security**
- `SECURITY.md` - Security policy
- `SECURITY_IMPLEMENTATION_GUIDE.md` - Security setup
- `.github/CODEOWNERS` - Access control

## 📋 File Purposes

| File | Purpose |
|------|---------|
| `app.py` | Main Flask application server |
| `config.py` | Project configuration and settings |
| `requirements.txt` | Standard Python dependencies |
| `requirements-gpu.txt` | GPU-optimized dependencies |
| `Procfile` | Heroku deployment configuration |
| `LICENSE` | MIT License for open source |
| `.gitignore` | Git ignore rules for clean repo |

## 🔄 Workflow

1. **Setup**: Follow `docs/INSTALLATION.md`
2. **Training**: Run `scripts/train.py` to train models
3. **Development**: Modify `app.py` and `modules/`
4. **Frontend**: Update `frontend/` files
5. **Security**: Review `.github/` and security docs

## ✨ Clean Structure Benefits

- **Clear Separation**: Each directory has a specific purpose
- **Easy Navigation**: Logical file organization
- **Professional**: Enterprise-grade structure
- **Maintainable**: Easy to find and modify files
- **Scalable**: Ready for future expansion

---

*This structure follows best practices for Python ML projects and ensures easy maintenance and development.*
