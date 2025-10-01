# AI Video Dubbing Application

This project allows you to dub videos using AI models. Follow the steps below to set up and run the application.

---

## Prerequisites

- Python 3.10 installed
- `pip` package manager
- CUDA-enabled GPU (optional, for faster PyTorch performance)
- API tokens for Hugging Face (`HF_TOKEN`) and Groq (`Groq_TOKEN`) if using their services

---

## All-in-One Setup

Open your terminal or command prompt and follow these steps:

```bash
# Step 1: Install required packages
pip install -r requirements.txt

# Step 2: Run additional installation script
python3.10 otherfileinstallation.py

# Step 3: Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 4: Create .env file with API keys
echo "# HF_TOKEN (required)" > .env
echo "HF_TOKEN=YOUR_HF_TOKEN_HERE" >> .env
echo "# Groq_TOKEN (optional)" >> .env
echo "Groq_TOKEN=YOUR_GROQ_TOKEN_HERE" >> .env

# Step 5: Run the video dubbing GUI
python3.10 video_dubbing_gui.py
