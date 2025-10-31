@echo off
echo ====================================
echo Setting up SMS Classification App
echo ====================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

echo Python found. Version:
python --version

REM Create virtual environment
echo.
echo Creating virtual environment...
if exist venv (
    echo Virtual environment already exists. Removing old one...
    rmdir /s /q venv
)
python -m venv venv

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo Installing requirements...
pip install -r requirements.txt

REM Download NLTK data (needed for text processing)
echo.
echo Downloading NLTK data...
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"

REM Check if installation was successful
echo.
echo Verifying installation...
python -c "import streamlit, pandas, plotly, numpy, sklearn, joblib, nltk, wordcloud, matplotlib, seaborn; print('All packages installed successfully!')"

if errorlevel 1 (
    echo.
    echo ERROR: Some packages failed to install
    pause
    exit /b 1
)

echo.
echo ====================================
echo Setup completed successfully!
echo ====================================
echo.
echo Starting Streamlit app...
echo Your app will open in your default browser at http://localhost:8501
echo.
echo To stop the app, press Ctrl+C in this window
echo.

REM Run the Streamlit app
streamlit run streamlit_app.py

pause