# 📁 FactTrack 2.0 - Clean Project Structure

## ✅ ORGANIZED & PROFESSIONAL

Your project is now cleanly organized into logical directories.

---

## 🗂️ DIRECTORY STRUCTURE

```
FACT-TRACK/
│
├── 📄 app.py                   # Main Flask application (run this!)
├── ⚙️ config.py                 # Central configuration
├── 📦 requirements.txt          # Dependencies
├── 📦 requirements-gpu.txt      # GPU dependencies
├── 🚀 Procfile                  # Deployment config
├── 📜 LICENSE                   # MIT License
├── 📖 README.md                 # Main documentation
├── 📁 STRUCTURE.md              # This file
│
├── 📂 scripts/                  # Utility scripts
│   ├── download_data.py        # Download training datasets
│   ├── train.py                # Train BERT models
│   └── test_installation.py   # Verify setup
│
├── 📂 modules/                  # ML modules
│   ├── __init__.py
│   ├── bert_category_model.py # BERT classifier (66M params)
│   ├── bert_bias_model.py     # BERT bias detector (fixed!)
│   ├── data_loader.py         # PyTorch datasets
│   ├── preprocess.py          # Text processing
│   └── utils.py               # Helper functions
│
├── 📂 frontend/                 # Web interface
│   ├── index.html              # Modern UI
│   ├── style.css               # Stunning design
│   └── script.js               # Interactive JS
│
├── 📂 docs/                     # Documentation
│   ├── INSTALL_GUIDE.md        # Installation help
│   ├── FIX_APPLIED.md          # Technical fixes
│   ├── COMMANDS.txt            # Quick reference
│   └── SOLUTION_NOW.txt        # Current status
│
├── 📂 data/                     # Datasets
│   ├── processed/              # Training data (created by scripts)
│   │   ├── category_data.csv  # 3000+ articles
│   │   └── bias_data.csv      # 2000+ balanced examples
│   ├── raw/                    # Downloaded datasets
│   ├── cache/                  # BERT model cache
│   └── sample_articles.csv     # Reference data
│
└── 📂 models/                   # Trained models
    ├── category_bert/          # Category model
    │   ├── model.pt           # Weights
    │   └── tokenizer files    # Tokenizer
    ├── bias_bert/             # Bias model
    │   ├── model.pt          # Weights
    │   └── tokenizer files   # Tokenizer
    ├── checkpoints/           # Best checkpoints
    ├── *.png                  # Confusion matrices & curves
    └── *.json                 # Performance metrics
```

---

## 📝 FILE COUNT SUMMARY

| Directory | Files | Purpose |
|-----------|-------|---------|
| **Root** | 6 | Core files only |
| **scripts/** | 3 | Training & utilities |
| **modules/** | 6 | ML models |
| **frontend/** | 3 | Web interface |
| **docs/** | 6 | Documentation |
| **data/** | varies | Datasets (created) |
| **models/** | varies | Trained models (created) |

**Total Core Files**: 24 essential files

---

## 🚀 QUICK COMMANDS

```bash
# Install
pip install -r requirements.txt

# Download data
python scripts/download_data.py

# Train models
python scripts/train.py

# Test setup
python scripts/test_installation.py

# Run app
python app.py

# Open browser
http://localhost:5000
```

---

## 📂 WHAT'S IN EACH DIRECTORY

### `/scripts` - Utility Scripts
- **download_data.py**: Downloads AG News and 20 Newsgroups, creates balanced bias dataset
- **train.py**: Fine-tunes BERT models, saves to `/models`
- **test_installation.py**: Verifies all dependencies and setup

### `/modules` - ML Modules
- **bert_category_model.py**: BERT classifier for 20+ categories
- **bert_bias_model.py**: BERT bias detector with all fixes
- **data_loader.py**: PyTorch Dataset and DataLoader creation
- **preprocess.py**: Text cleaning and paragraph splitting
- **utils.py**: Metrics, plotting, balancing, helpers

### `/frontend` - Web Interface
- **index.html**: Modern UI with animations and SVG icons
- **style.css**: Dark theme, glassmorphism, gradients
- **script.js**: API integration, dynamic rendering

### `/docs` - Documentation
- **INSTALL_GUIDE.md**: Troubleshooting installation issues
- **FIX_APPLIED.md**: PyTorch 2.6+ compatibility fix explanation
- **COMMANDS.txt**: Quick command reference card
- **SOLUTION_NOW.txt**: Current status and next steps

### `/data` - Datasets (Created by Scripts)
- **processed/**: Cleaned training data ready for BERT
- **raw/**: Original downloaded datasets
- **cache/**: Hugging Face model cache

### `/models` - Trained Models (Created by Training)
- **category_bert/**: Fine-tuned DistilBERT for categories
- **bias_bert/**: Fine-tuned DistilBERT for bias
- **checkpoints/**: Best model checkpoints during training
- **Visualizations**: Confusion matrices and training curves

---

## 🎯 BENEFITS OF THIS STRUCTURE

### Clean Root Directory
✅ Only 6 essential files in root
✅ No clutter
✅ Professional appearance
✅ Easy to navigate

### Logical Organization
✅ Scripts grouped together
✅ Documentation in `/docs`
✅ ML code in `/modules`
✅ UI in `/frontend`
✅ Clear separation of concerns

### Easy to Find
✅ Need to train? → `scripts/train.py`
✅ Need docs? → `docs/`
✅ Need to configure? → `config.py`
✅ Need to run? → `app.py`

### Professional
✅ Follows Python project conventions
✅ Scalable structure
✅ Easy to add new features
✅ Clear file purposes

---

## 📖 NAVIGATION GUIDE

**Want to...**

- **Run the app?** → `app.py`
- **Configure settings?** → `config.py`
- **Train models?** → `scripts/train.py`
- **Download data?** → `scripts/download_data.py`
- **Edit UI?** → `frontend/`
- **Modify ML models?** → `modules/`
- **Read docs?** → `docs/` or `README.md`
- **Check requirements?** → `requirements.txt`

---

## 🎊 NEXT STEPS

Since you already have trained models, just run:

```bash
python app.py
```

Open: `http://localhost:5000`

---

**Clean • Organized • Professional** ✨

*Version 2.0.0 - Production-Ready Architecture*
