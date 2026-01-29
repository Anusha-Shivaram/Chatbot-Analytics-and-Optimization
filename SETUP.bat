@echo off
echo ========================================
echo Chatbot Analytics - Quick Setup
echo ========================================
echo.

echo Installing packages...
python -m pip install pandas numpy matplotlib seaborn scikit-learn plotly dash dash-bootstrap-components nltk textblob wordcloud scipy openpyxl --quiet --disable-pip-version-check

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Package installation failed
    echo Make sure Python is installed and in PATH
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Run the analysis with:
echo   python run_analysis.py
echo.
pause
