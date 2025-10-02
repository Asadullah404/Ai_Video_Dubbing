# AI Video Dubbing Application 🎬🤖

[![Python](https://img.shields.io/badge/python-3.10.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13-orange?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

An AI-powered video dubbing application that enables automatic lip-sync and audio dubbing using state-of-the-art machine learning models. Transform videos into different languages while maintaining natural lip movements and audio quality.

## 🌟 About

This project is built upon and modified from the excellent [ViDubb project by medahmedkrichen](https://github.com/medahmedkrichen/ViDubb). I am deeply grateful to the original creators for their groundbreaking work and for making this technology accessible to the community.

## ✨ Features

- **AI-Powered Lip Synchronization**: Automatically sync lip movements with dubbed audio
- **Multi-language Support**: Dub videos into different languages using advanced TTS
- **GPU Acceleration**: Optional CUDA support for faster processing
- **User-Friendly GUI**: Intuitive interface for easy video processing
- **High-Quality Output**: Maintains video quality while generating natural-looking results

## 📋 Prerequisites

Before you begin, ensure your system meets these requirements:

- **Python 3.10.11** (recommended for optimal compatibility)
- **pip** package manager
- **CUDA-enabled GPU** (optional, but recommended for faster processing)
- **FFmpeg** (required for video/audio processing)
- **Microsoft Visual Studio with C++ build tools** (Windows only, required for building Python wheels)
- **API Tokens**:
  - Hugging Face token (`HF_TOKEN`) - Required
  - Groq token (`Groq_TOKEN`) - Optional

## 🚀 Installation

Follow these steps carefully to set up the application:

### 1. Clone the Repository

```bash
git clone <[your-repository-url](https://github.com/Asadullah404/Ai_Video_Dubbing)>
cd ai-video-dubbing
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Additional Setup Script

```bash
python3.10 otherfileinstallation.py
```

### 4. Install PyTorch with CUDA Support

For GPU acceleration (CUDA 12.1):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only installation:

```bash
pip install torch torchvision torchaudio
```

### 5. Download Pre-trained Models

Ensure the `Wav2Lip` folder exists, then download the required models:

```bash
# Wav2Lip GAN model
wget 'https://github.com/medahmedkrichen/ViDubb/releases/download/weights2/wav2lip_gan.1.1.pth' -O 'Wav2Lip/wav2lip_gan.pth'

# Face detection model (S3FD)
wget 'https://github.com/medahmedkrichen/ViDubb/releases/download/weights1/s3fd-619a316812.1.1.pth' -O 'Wav2Lip/face_detection/detection/sfd/s3fd.pth'
```

**Windows users** can use PowerShell or download manually from the URLs above.

### 6. Configure API Keys

Create a `.env` file in the project root directory:

```env
# Required: Hugging Face API token
HF_TOKEN=your_hugging_face_token_here

# Optional: Groq API token
Groq_TOKEN=your_groq_token_here
```

To obtain API tokens:
- **Hugging Face**: Sign up at [huggingface.co](https://huggingface.co) and generate a token from your profile settings
- **Groq**: Visit [groq.com](https://groq.com) to create an account and obtain an API key

### 7. Install FFmpeg

**Windows:**
1. Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract the archive
3. Add the `bin` folder to your system PATH

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

Verify installation:
```bash
ffmpeg -version
```

## 🎯 Usage

Launch the application GUI:

```bash
python3.10 video_dubbing_gui.py
```

The graphical interface will guide you through:
1. Loading your source video
2. Selecting target language and voice options
3. Processing the video with AI lip-sync
4. Exporting the dubbed result

## 🛠️ Troubleshooting

### Common Issues

**Missing `cudart64_*.dll` errors:**
- Ensure your CUDA version matches the PyTorch installation
- Verify CUDA is properly installed and added to PATH

**Module not found errors:**
- Confirm you're using Python 3.10.11
- Check that all dependencies installed successfully
- Try creating a fresh virtual environment

**FFmpeg not found:**
- Verify FFmpeg is installed: `ffmpeg -version`
- Ensure FFmpeg is added to your system PATH
- Restart your terminal after PATH modifications

**TTS compilation errors (Windows):**
- Install Visual Studio with C++ build tools
- Ensure the Windows SDK is installed

**GPU not being utilized:**
- Verify CUDA installation: `nvidia-smi`
- Check PyTorch CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`

## 📁 Project Structure

```
ai-video-dubbing/
├── Wav2Lip/                  # Lip-sync model files
│   ├── wav2lip_gan.pth      # Pre-trained weights
│   └── face_detection/       # Face detection models
├── video_dubbing_gui.py      # Main GUI application
├── requirements.txt          # Python dependencies
├── otherfileinstallation.py  # Additional setup script
├── .env                      # API configuration (create this)
└── README.md                 # This file
```

## 🙏 Acknowledgments

This project builds upon the exceptional work of:
- **[ViDubb](https://github.com/medahmedkrichen/ViDubb)** by medahmedkrichen - The foundation of this application
- **Wav2Lip** researchers for the lip-sync technology
- The open-source AI community for making these tools accessible

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit a pull request.

## ⚠️ Important Notes

- **Python Version**: Strictly use Python 3.10.11 for maximum compatibility
- **Model Downloads**: Ensure models are downloaded to the correct directories
- **API Limits**: Be aware of rate limits on Hugging Face and Groq APIs
- **Processing Time**: Video processing can be time-intensive, especially without GPU acceleration

## 📧 Support

If you encounter issues not covered in the troubleshooting section, please:
1. Check existing GitHub issues
2. Review the original [ViDubb documentation](https://github.com/medahmedkrichen/ViDubb)
3. Open a new issue with detailed error logs

---


