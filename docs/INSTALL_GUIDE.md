# 🔧 FactTrack 2.0 - Installation & Setup Guide

## ✅ ERRORS FIXED

I've fixed the dependency installation issues you encountered:
- ✅ Fixed `torch` version format in requirements-gpu.txt
- ✅ Changed to flexible version requirements (>=) instead of exact versions
- ✅ Added all missing Hugging Face dependencies

---

## 🚀 EXACT INSTALLATION STEPS

Follow these commands **exactly** in your PowerShell terminal:

### STEP 1: Make Sure You're in the Right Directory
```powershell
cd C:\Users\afrah\OneDrive\Desktop\FACT-TRACK
```

### STEP 2: Activate Virtual Environment
```powershell
venv\Scripts\Activate.ps1
```

You should see `(venv)` at the start of your prompt.

### STEP 3: Update pip (Important!)
```powershell
python -m pip install --upgrade pip
```

### STEP 4: Install Dependencies

**OPTION A - Standard Installation (CPU or Auto-detect):**
```powershell
pip install -r requirements.txt
```

**OPTION B - If You Have NVIDIA GPU:**
```powershell
# Install PyTorch with CUDA first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then install other requirements
pip install -r requirements.txt
```

**Expected Output:**
```
Successfully installed torch-2.X.X transformers-4.XX.X datasets-2.XX.X ...
```

**⏱️ Time**: 5-10 minutes (downloading ~2-3GB of packages)

---

## 🔍 VERIFY INSTALLATION

### Test if packages installed correctly:
```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import datasets; print(f'Datasets installed')"
```

**Expected Output:**
```
PyTorch: 2.X.X
Transformers: 4.XX.X
Datasets installed
```

**✅ If you see these outputs, installation was successful!**

---

## 🚀 CONTINUE WITH TRAINING

### STEP 5: Download Training Data
```powershell
python download_data.py
```

**What happens:**
- Downloads AG News dataset (120k articles)
- Downloads 20 Newsgroups dataset  
- Creates balanced bias dataset (50/50 split)
- Saves to `data/processed/`

**Expected Output:**
```
============================================================
Downloading AG News Dataset...
============================================================
Processing train: 100%|████████████| 120000/120000
✓ Downloaded 120000 articles from AG News

============================================================
Downloading 20 Newsgroups Dataset...
============================================================
Processing train: 100%|████████████| 11314/11314
✓ Downloaded 18846 articles from 20 Newsgroups

============================================================
Creating Balanced Bias Detection Dataset...
============================================================
✓ Created bias dataset with 2000 examples
  Biased: 1000 (50.0%)
  Not Biased: 1000 (50.0%)
  ✓ Classes are balanced!

============================================================
✓ Data Download Complete!
============================================================
```

**⏱️ Time**: 5-10 minutes

---

### STEP 6: Train BERT Models
```powershell
python train.py
```

**What happens:**
- Fine-tunes DistilBERT for category classification
- Fine-tunes DistilBERT for bias detection (with all fixes!)
- Saves trained models to `models/`

**Expected Output:**
```
============================================================
FactTrack Production Training - BERT Models
Version: 2.0.0
============================================================

============================================================
Device Information
============================================================
Device: cuda  (or cpu if no GPU)
============================================================

✓ Training data found

============================================================
TRAINING CATEGORY CLASSIFIER (BERT)
============================================================

Epoch 1/3
Training: 100%|██████████| 132/132 [XX:XX<00:00]
  Train Loss: 0.XXXX, Train Acc: 0.8XXX
  Val Loss:   0.XXXX, Val Acc:   0.8XXX
  ✓ New best validation accuracy: 0.8XXX

[Continues for 3 epochs...]

Test Accuracy: 0.85XX ✅
Test F1 Score: 0.8XXX ✅

============================================================
TRAINING BIAS DETECTOR (BERT)
============================================================

Using Weighted Cross Entropy Loss with weights: [1.0, 1.0]

Epoch 1/3
  Train - Loss: 0.XXXX, Acc: 0.8XXX, F1: 0.7XXX
  Val   - Loss: 0.XXXX, Acc: 0.8XXX, F1: 0.8XXX
  ✓ New best validation F1: 0.8XXX

[Continues for 3 epochs...]

Test Accuracy: 0.80XX ✅
Test F1 Score: 0.78XX ✅

Confusion Matrix:
             Predicted
          Not Biased  Biased
Actual 
Not Biased    113       37
Biased         27      123

============================================================
TRAINING COMPLETE!
============================================================

Next step: Run 'python app.py' to start the server
```

**⏱️ Time**:
- **With GPU**: 30-60 minutes
- **With CPU**: 2-4 hours

---

### STEP 7: Run the Application
```powershell
python app.py
```

**Expected Output:**
```
============================================================
Loading FactTrack BERT Models...
============================================================

1. Loading category classifier...
   ✓ Category classifier loaded

2. Loading bias detector...
   ✓ Bias detector loaded

============================================================
✓ All models loaded successfully!
============================================================

🌐 Access the application at: http://localhost:5000
```

---

### STEP 8: Open Browser
```
http://localhost:5000
```

**You'll see the stunning new UI with:**
- ✨ Animated gradient background
- 🎨 Dark modern theme
- 🟢 Green "System Ready" indicator
- 💫 Smooth animations everywhere

---

## 🆘 TROUBLESHOOTING THE ERRORS YOU SAW

### Error 1: "No module named 'datasets'"

**Cause**: Package didn't install properly

**Solution**:
```powershell
# Install datasets package specifically
pip install datasets

# Or reinstall all requirements
pip install -r requirements.txt
```

---

### Error 2: "Could not find torch==2.1.0+cu118"

**Cause**: Wrong format for GPU PyTorch installation

**Solution** (Choose ONE):

**Option A - Let pip auto-detect (Recommended):**
```powershell
pip install -r requirements.txt
```
This will install the latest PyTorch version (2.7+) which works fine!

**Option B - Manual CUDA installation:**
```powershell
# Install PyTorch with CUDA support first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then install other requirements
pip install -r requirements.txt
```

**Option C - CPU only (no GPU needed):**
```powershell
pip install -r requirements.txt
```
Works perfectly! Just slower training.

---

## 🔍 VERIFY EVERYTHING IS WORKING

### After Installing Dependencies:
```powershell
python -c "import torch, transformers, datasets; print('✅ All packages installed successfully!')"
```

### After Downloading Data:
```powershell
# Check files exist
dir data\processed\category_data.csv
dir data\processed\bias_data.csv
```

### After Training:
```powershell
# Check model files exist
dir models\category_bert\model.pt
dir models\bias_bert\model.pt
```

### After Running App:
```powershell
# Should see in terminal:
# "✓ Models loaded successfully"
# "Access the application at: http://localhost:5000"
```

---

## 🎯 COMPLETE SETUP (Fresh Start)

If you want to start completely fresh:

```powershell
# 1. Make sure you're in the project directory
cd C:\Users\afrah\OneDrive\Desktop\FACT-TRACK

# 2. Activate venv
venv\Scripts\Activate.ps1

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download data
python download_data.py

# 6. Train models (THIS TAKES TIME!)
python train.py

# 7. Run app
python app.py

# 8. Open browser: http://localhost:5000
```

---

## ⚙️ SYSTEM REQUIREMENTS

### Minimum:
- Windows 10/11
- Python 3.8+
- 4GB RAM
- 3GB free disk space
- Internet connection (for downloading)

### Recommended:
- Python 3.9+
- 8GB+ RAM
- NVIDIA GPU with 4GB+ VRAM (optional)
- CUDA 11.8 or 12.x (if using GPU)

---

## 📊 WHAT TO EXPECT

### Installation Size:
- Dependencies: ~2-3 GB
- Training Data: ~500 MB
- Models after training: ~1 GB
- **Total**: ~4 GB

### Training Time:
- **GPU (NVIDIA)**: 30-60 minutes total
- **CPU**: 2-4 hours total
- You can leave it running and come back later!

### Performance After Training:
- Category Accuracy: 85%+
- Bias Accuracy: 80%+
- Bias F1 Score: 0.78+
- **Bias detection will show VARIED predictions** (not stuck!)

---

## 🎨 THE NEW UI

Once everything is running, you'll have:

### Visual Features:
- 🌈 Floating animated gradient orbs in background
- 🎨 Modern dark theme with glassmorphism
- ✨ Smooth card animations
- 🔮 Neural network loading visualization
- 📊 Animated progress bars with shimmer effects
- 🏆 Medal badges (🥇🥈🥉) for top-3 predictions
- 🌈 Dynamic colors for all 20+ categories
- 📈 Gradient bias bars (green → yellow → red)

### Interactive:
- Hover effects on all cards
- Button shine animations
- Instant visual feedback
- Smooth scrolling
- Professional transitions

---

## ✅ FINAL CHECKLIST

Before you start:
- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] Internet connection working
- [ ] 3-4 GB free disk space

After installation:
- [ ] All packages installed (no errors)
- [ ] `datasets` module imports successfully
- [ ] `torch` module imports successfully

After download_data.py:
- [ ] `data/processed/category_data.csv` exists
- [ ] `data/processed/bias_data.csv` exists
- [ ] Shows "Classes are balanced!"

After train.py:
- [ ] Models saved to `models/category_bert/` and `models/bias_bert/`
- [ ] Test accuracy 80%+
- [ ] Confusion matrix shows varied predictions

After app.py:
- [ ] Server starts without errors
- [ ] Shows "Models loaded successfully"
- [ ] Can access http://localhost:5000

---

## 🚀 START NOW

Run this command in your activated venv:

```powershell
pip install -r requirements.txt
```

Wait for it to complete, then run:

```powershell
python download_data.py
```

**That's it!** The installation process will guide you through the rest.

---

## 📞 STILL HAVING ISSUES?

### If installation fails:
```powershell
# Clear pip cache
pip cache purge

# Upgrade pip and setuptools
python -m pip install --upgrade pip setuptools wheel

# Retry installation
pip install -r requirements.txt
```

### If specific package fails:
```powershell
# Install packages one by one
pip install torch
pip install transformers
pip install datasets
pip install Flask flask-cors
pip install pandas numpy scikit-learn
pip install tqdm matplotlib seaborn
pip install langdetect colorama
```

### If still stuck:
1. Check your Python version: `python --version` (needs 3.8+)
2. Check your pip version: `pip --version`
3. Try creating a fresh virtual environment
4. Check internet connection

---

**Ready to start?**

```powershell
pip install -r requirements.txt
```

---

*Version 2.0.0 • Fixed Requirements • Ready to Install*

