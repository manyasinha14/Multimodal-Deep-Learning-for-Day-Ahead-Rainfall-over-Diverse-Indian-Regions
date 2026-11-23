#!/usr/bin/env bash
# Start the Monsoon Rainfall Portal on macOS / Linux.
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
else
  source .venv/bin/activate
fi
streamlit run app.py
