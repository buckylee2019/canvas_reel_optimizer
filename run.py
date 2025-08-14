#!/usr/bin/env python3
"""
Simple runner script for Nova Canvas & Reel Optimizer
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if requirements are installed"""
    try:
        import streamlit
        import boto3
        import PIL
        print("✅ All requirements are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing requirement: {e}")
        print("Please run: python setup.py")
        return False

def check_directories():
    """Ensure output directories exist"""
    directories = ["generated_images", "generated_videos", "shot_images"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

def main():
    """Main runner function"""
    print("🎬 Starting Nova Canvas & Reel Optimizer...")
    
    # Check if requirements are installed
    if not check_requirements():
        sys.exit(1)
    
    # Create directories
    check_directories()
    
    # Check if streamlit_app.py exists
    if not Path("streamlit_app.py").exists():
        print("❌ streamlit_app.py not found in current directory")
        sys.exit(1)
    
    # Start Streamlit
    print("🚀 Starting Streamlit application...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

if __name__ == "__main__":
    main()
