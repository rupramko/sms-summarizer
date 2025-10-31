# SMS Classification App - Installation Guide

## Quick Start (Recommended)

### Option 1: Use the automated script (Windows)
1. **Run the batch file:**
   ```
   Double-click on install_and_run.bat
   ```

2. **Or run the PowerShell script:**
   ```powershell
   .\install_and_run.ps1
   ```

### Option 2: Manual Installation

1. **Create and activate virtual environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Upgrade pip:**
   ```powershell
   python -m pip install --upgrade pip
   ```

3. **Install requirements:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Download NLTK data:**
   ```powershell
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
   ```

5. **Run the app:**
   ```powershell
   streamlit run streamlit_app.py
   ```

## Package Details

The requirements.txt installs:
- **streamlit** (≥1.28.0) - Web framework for the app
- **pandas** (≥2.0.0) - Data manipulation
- **plotly** (≥5.15.0) - Interactive visualizations
- **numpy** (≥1.24.0) - Numerical operations
- **scikit-learn** (≥1.3.0) - Machine learning algorithms
- **joblib** (≥1.3.0) - Model serialization
- **nltk** (≥3.8.0) - Natural language processing
- **wordcloud** (≥1.9.0) - Word cloud generation
- **matplotlib** (≥3.7.0) - Plotting
- **seaborn** (≥0.12.0) - Statistical visualizations

## Troubleshooting

### If you get execution policy errors in PowerShell:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### If pip install fails:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### If NLTK downloads fail:
```powershell
python -c "import ssl; ssl._create_default_https_context = ssl._create_unverified_context; import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
```

## Running the App

After successful installation, the app will be available at:
**http://localhost:8501**

To stop the app, press `Ctrl+C` in the terminal.

## Files in this project:
- `streamlit_app.py` - Main Streamlit application
- `unsupervised_sms_classifier.py` - SMS classification logic
- `requirements.txt` - Python dependencies
- `install_and_run.bat` - Windows batch setup script
- `install_and_run.ps1` - PowerShell setup script