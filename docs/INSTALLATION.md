# 🔧 FactTrack Installation Guide

Complete installation and setup guide for FACT-TRACK.

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download & Train Models
```bash
# Download datasets (5-10 min)
python scripts/download_data.py

# Train BERT models (30min-4hrs)
python scripts/train.py
```

### Step 3: Run Application
```bash
python app.py
# Open: http://localhost:5000
```

---

## 📋 Detailed Installation

### Prerequisites
- Python 3.8+
- 4GB RAM minimum
- 3GB free disk space
- Internet connection

### Virtual Environment Setup

**Windows:**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

**Standard Installation:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**GPU Installation (NVIDIA):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Verify Installation
```bash
python -c "import torch, transformers, datasets; print('✅ All packages installed!')"
```

---

## 🎯 Training Process

### Download Training Data
```bash
python scripts/download_data.py
```
- Downloads AG News dataset (120k articles)
- Downloads 20 Newsgroups dataset
- Creates balanced bias dataset (50/50 split)

### Train Models
```bash
python scripts/train.py
```
- Fine-tunes DistilBERT for category classification
- Fine-tunes DistilBERT for bias detection
- Saves models to `models/` directory

**Training Time:**
- GPU: 30-60 minutes
- CPU: 2-4 hours

---

## 🆘 Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| "No module named torch" | Activate venv: `venv\Scripts\Activate.ps1` |
| "Could not find torch==2.1.0+cu118" | Use `pip install -r requirements.txt` |
| "Data not found" | Run `python scripts/download_data.py` |
| Out of memory | Edit `config.py`: `BATCH_SIZE = 4` |
| Training too slow | Use GPU or reduce `EPOCHS = 2` |

### Package Installation Issues
```bash
# Clear pip cache
pip cache purge

# Upgrade pip and setuptools
python -m pip install --upgrade pip setuptools wheel

# Install packages individually if needed
pip install torch
pip install transformers
pip install datasets
pip install Flask flask-cors
```

---

## ✅ Verification Checklist

After installation:
- [ ] All packages installed (no errors)
- [ ] `data/processed/category_data.csv` exists
- [ ] `data/processed/bias_data.csv` exists
- [ ] Models saved to `models/category_bert/` and `models/bias_bert/`
- [ ] Test accuracy 80%+
- [ ] Server starts without errors
- [ ] Can access http://localhost:5000

---

## 📊 Expected Results

### Performance Metrics
- Category Accuracy: 85%+
- Bias Accuracy: 80%+
- Bias F1 Score: 0.78+

### File Sizes
- Dependencies: ~2-3 GB
- Training Data: ~500 MB
- Models: ~1 GB
- **Total**: ~4 GB

---

*For more help, see the main [README.md](../README.md)*
