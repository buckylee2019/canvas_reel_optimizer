#!/usr/bin/env python3
"""
Setup script for Nova Canvas & Reel Optimizer
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.10+"""
    if sys.version_info < (3, 10):
        print("❌ Error: Python 3.10 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        sys.exit(1)

def create_directories():
    """Create necessary directories"""
    directories = ["generated_images", "generated_videos", "shot_images"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")

def check_aws_credentials():
    """Check if AWS credentials are configured"""
    try:
        import boto3
        session = boto3.Session()
        credentials = session.get_credentials()
        if credentials is None:
            print("⚠️  Warning: AWS credentials not found")
            print("Please configure AWS credentials using:")
            print("  aws configure")
            print("  or set environment variables:")
            print("  export AWS_ACCESS_KEY_ID=your_access_key")
            print("  export AWS_SECRET_ACCESS_KEY=your_secret_key")
            print("  export AWS_DEFAULT_REGION=us-east-1")
        else:
            print("✅ AWS credentials found")
    except ImportError:
        print("⚠️  boto3 not installed yet, will check AWS credentials after installation")

def main():
    """Main setup function"""
    print("🎬 Nova Canvas & Reel Optimizer Setup")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Check AWS credentials (before installing boto3)
    check_aws_credentials()
    
    # Install requirements
    install_requirements()
    
    # Create directories
    create_directories()
    
    # Final check for AWS credentials
    check_aws_credentials()
    
    print("\n🎉 Setup completed successfully!")
    print("\n🚀 To start the application:")
    print("  streamlit run streamlit_app.py")
    print("\n📖 For more information, see README.md")
    print("\n⚙️  Don't forget to:")
    print("  1. Configure your AWS credentials")
    print("  2. Update your S3 bucket name in config.py")
    print("  3. Ensure you have access to Amazon Bedrock models")

if __name__ == "__main__":
    main()
