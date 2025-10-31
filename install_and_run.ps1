# SMS Classification App Setup and Run Script
Write-Host "====================================" -ForegroundColor Green
Write-Host "Setting up SMS Classification App" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python from https://python.org" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Create virtual environment
Write-Host "`nCreating virtual environment..." -ForegroundColor Cyan
if (Test-Path "venv") {
    Write-Host "Virtual environment already exists. Removing old one..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "venv"
}
python -m venv venv

# Activate virtual environment
Write-Host "`nActivating virtual environment..." -ForegroundColor Cyan
& ".\venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Install requirements
Write-Host "`nInstalling requirements..." -ForegroundColor Cyan
pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install requirements" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Download NLTK data
Write-Host "`nDownloading NLTK data..." -ForegroundColor Cyan
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"

# Verify installation
Write-Host "`nVerifying installation..." -ForegroundColor Cyan
try {
    python -c "import streamlit, pandas, plotly, numpy, sklearn, joblib, nltk, wordcloud, matplotlib, seaborn; print('All packages installed successfully!')"
    Write-Host "All packages verified successfully!" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Some packages failed to install properly" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "`n====================================" -ForegroundColor Green
Write-Host "Setup completed successfully!" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Write-Host "`nStarting Streamlit app..." -ForegroundColor Cyan
Write-Host "Your app will open in your default browser at http://localhost:8501" -ForegroundColor Yellow
Write-Host "`nTo stop the app, press Ctrl+C in this window" -ForegroundColor Yellow

# Run the Streamlit app
streamlit run streamlit_app.py