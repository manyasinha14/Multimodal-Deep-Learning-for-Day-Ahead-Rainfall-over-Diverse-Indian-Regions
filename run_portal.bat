@echo off
REM Double-click to start the Monsoon Rainfall Portal on Windows.
cd /d "%~dp0"
if not exist .venv (
    echo Creating a virtual environment - first run only, this takes a few minutes...
    python -m venv .venv
    call .venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate
)
streamlit run app.py
pause
