#!/usr/bin/env python3
"""
run_app.py
One-click launcher for the Dawah Dialectics Web Application & Q&A Assistant.
Usage: python run_app.py
"""

import os
import sys
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import uvicorn
from web.app import app

if __name__ == "__main__":
    port = 8000
    url = f"http://127.0.0.1:{port}"
    print("=" * 70)
    print("  🕌  DAWAH DIALECTICS: ISLAMIC APOLOGETICS Q&A ENGINE")
    print(f"  🌐  Web App running at: {url}")
    print("  ⚡  Zero-LLM instant smart assistant enabled.")
    print("  📖  Bilingual UI (English & العربية) ready.")
    print("=" * 70)
    
    # Try opening browser automatically
    try:
        webbrowser.open(url)
    except Exception:
        pass

    uvicorn.run("web.app:app", host="127.0.0.1", port=port, reload=True)
