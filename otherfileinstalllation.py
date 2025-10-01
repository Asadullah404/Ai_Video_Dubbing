#!/usr/bin/env python3
"""
Dependency Installation Script for Advanced Video Dubbing System
This script installs all required dependencies for the video dubbing system.
"""

import os
import platform
import subprocess
import sys

print("🚀 Advanced Video Dubbing System - Dependency Installer")
print("=" * 60)

def get_silent_redirect():
    """Get platform-specific silent redirect"""
    return ">nul 2>&1" if platform.system() == 'Windows' else "> /dev/null 2>&1"

def install_package(import_name, install_command, description=""):
    """Install a package if not already installed"""
    try:
        __import__(import_name)
        print(f"✓ {description or import_name} already installed")
        return True
    except ImportError:
        print(f"📦 Installing {description or import_name}...")
        silent = get_silent_redirect()
        result = os.system(f"{install_command} {silent}")
        if result == 0:
            print(f"✓ {description or import_name} installed successfully")
            return True
        else:
            print(f"✗ Failed to install {description or import_name}")
            return False

def main():
    silent = get_silent_redirect()
    failed_packages = []
    
    print("\n📋 Installing Core Dependencies...")
    print("-" * 60)
    
    # Core dependencies with specific versions
    packages = [
        ('protobuf', 'pip install protobuf==3.20.3', 'Protocol Buffers'),
        ('spacy', 'pip install spacy==3.8.2', 'spaCy NLP'),
        ('TTS', 'pip install --no-deps TTS==0.21.0', 'Coqui TTS'),
        ('packaging', 'pip install packaging==20.9', 'Packaging'),
        ('whisper', 'pip install openai-whisper==20240930', 'OpenAI Whisper'),
        ('deepface', 'pip install deepface==0.0.93', 'DeepFace'),
        ('gtts', 'pip install gtts', 'Google Text-to-Speech'),
        ('pydub', 'pip install pydub', 'PyDub Audio Processing'),
        ('pedalboard', 'pip install pedalboard', 'Pedalboard Audio Effects'),
        ('noisereduce', 'pip install noisereduce', 'Noise Reduction'),
        ('resemblyzer', 'pip install resemblyzer', 'Voice Encoder'),
        ('IPython', 'pip install ipython', 'IPython Interactive Shell'),
    ]
    
    for import_name, install_cmd, desc in packages:
        if not install_package(import_name, install_cmd, desc):
            failed_packages.append(desc)
    
    # NumPy - force specific version
    print("\n📦 Installing NumPy 1.26.4...")
    os.system(f'pip install numpy==1.26.4 {silent}')
    print("✓ NumPy installed")
    
    # Additional dependencies (installed via pip automatically with above packages)
    print("\n📋 Installing Additional Dependencies...")
    print("-" * 60)
    
    additional_packages = [
        ('pyannote.audio', 'pip install pyannote.audio', 'Pyannote Audio'),
        ('audio_separator', 'pip install audio-separator', 'Audio Separator'),
        ('transformers', 'pip install transformers', 'Hugging Face Transformers'),
        ('torch', 'pip install torch torchvision torchaudio', 'PyTorch'),
        ('speechbrain', 'pip install speechbrain', 'SpeechBrain'),
        ('cv2', 'pip install opencv-python', 'OpenCV'),
        ('groq', 'pip install groq', 'Groq API'),
        ('scipy', 'pip install scipy', 'SciPy'),
        ('ffmpeg', 'pip install ffmpeg-python', 'FFmpeg Python'),
        ('dotenv', 'pip install python-dotenv', 'Python Dotenv'),
        ('nltk', 'pip install nltk', 'NLTK'),
        ('faster_whisper', 'pip install faster-whisper', 'Faster Whisper'),
        ('librosa', 'pip install librosa', 'Librosa Audio Analysis'),
        ('soundfile', 'pip install soundfile', 'Sound File I/O'),
        ('ascii_magic', 'pip install ascii-magic', 'ASCII Art'),
        ('yt_dlp', 'pip install yt-dlp', 'YouTube Downloader'),
    ]
    
    for import_name, install_cmd, desc in additional_packages:
        if not install_package(import_name, install_cmd, desc):
            failed_packages.append(desc)
    
    # Download NLTK data
    print("\n📥 Downloading NLTK Data...")
    print("-" * 60)
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        print("✓ NLTK data downloaded")
    except Exception as e:
        print(f"⚠ NLTK data download failed: {e}")
        failed_packages.append("NLTK Data")
    
    # System requirements check
    print("\n🔍 Checking System Requirements...")
    print("-" * 60)
    
    # Check FFmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              timeout=5)
        if result.returncode == 0:
            print("✓ FFmpeg is installed")
        else:
            print("⚠ FFmpeg not found - required for video processing")
            print("  Install from: https://ffmpeg.org/download.html")
    except:
        print("⚠ FFmpeg not found - required for video processing")
        print("  Install from: https://ffmpeg.org/download.html")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 INSTALLATION SUMMARY")
    print("=" * 60)
    
    if failed_packages:
        print(f"\n⚠ {len(failed_packages)} package(s) failed to install:")
        for pkg in failed_packages:
            print(f"  ✗ {pkg}")
        print("\n💡 Try installing failed packages manually:")
        print("   pip install <package-name>")
    else:
        print("\n✅ All dependencies installed successfully!")
    
    print("\n📝 NOTES:")
    print("-" * 60)
    print("1. Ensure you have FFmpeg installed on your system")
    print("2. Set up environment variables in .env file:")
    print("   - HF_TOKEN: Hugging Face token for speaker diarization")
    print("   - Groq_TOKEN: Groq API token for context translation")
    print("3. For Wav2Lip lip-sync, clone the repository separately:")
    print("   git clone https://github.com/Rudrabha/Wav2Lip.git")
    print("4. GPU support requires CUDA-compatible PyTorch installation")
    print("\n🎬 You're ready to run the video dubbing system!")
    print("=" * 60)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Installation error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)