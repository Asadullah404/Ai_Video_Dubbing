# 🚀 How to Run AI Video Dubbing on Google Colab & Kaggle GPUs

You can offload all heavy processing (Whisper Large-v3, XTTS v2 Voice Cloning, Wav2Lip GAN Lip-Sync, and UVR Vocals Isolation) to **free Cloud GPUs** on Google Colab (NVIDIA Tesla T4 / A100) or Kaggle (Dual Tesla T4 / P100).

---

## ⚡ Option 1: Google Colab (Recommended)

### Step 1: Open Google Colab
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **Upload** and upload the included [`colab_gpu_server.ipynb`](file:///D:/video_dub/Ai_Video_Dubbing-main/Ai_Video_Dubbing-main/colab_gpu_server.ipynb) notebook.
3. In the top menu, go to **Runtime > Change runtime type** and choose **T4 GPU** (or A100 if you have Colab Pro).

### Step 2: Run the Notebook
1. Run **Cell 1** to install the CUDA PyTorch environment, `coqui-tts`, and pre-trained models.
2. Run **Cell 2** to start the GPU Server & Cloudflare Tunnel.

Colab will output a public URL like:
```text
*****************************************************************
🎉 YOUR CLOUD GPU SERVER IS READY!
🔗 Public URL: https://example-cloud-gpu.trycloudflare.com
👉 Copy and paste this URL into your Local Dubbing GUI.
*****************************************************************
```

---

## ⚡ Option 2: Kaggle Notebooks (Dual T4 / P100 GPUs)

### Step 1: Create a Kaggle Notebook
1. Go to [kaggle.com/code](https://www.kaggle.com/code) and click **New Notebook**.
2. In the right-hand panel under **Settings**:
   - **Accelerator**: Select **GPU T4 x2** or **GPU P100**.
   - **Internet**: Toggle to **On** (required for downloads and tunneling).

### Step 2: Paste the Setup Code
In the first cell, paste and run:
```bash
!git clone https://github.com/Asadullah404/Ai_Video_Dubbing.git dubbing_app || (cd dubbing_app && git pull)
%cd dubbing_app
!pip install -q fastapi uvicorn python-multipart pycloudflared nest-asyncio deep-translator
!pip install -q faster-whisper coqui-tts pyannote.audio audio-separator[gpu] speechbrain groq librosa soundfile noisereduce pedalboard yt-dlp opencv-python

!mkdir -p Wav2Lip/face_detection/detection/sfd
!wget -q -c 'https://github.com/medahmedkrichen/ViDubb/releases/download/weights2/wav2lip_gan.1.1.pth' -O 'Wav2Lip/wav2lip_gan.pth'
!wget -q -c 'https://github.com/medahmedkrichen/ViDubb/releases/download/weights1/s3fd-619a316812.1.1.pth' -O 'Wav2Lip/face_detection/detection/sfd/s3fd.pth'

!python colab_server.py
```

---

## 🖥️ Step 3: Connect Your Local GUI to the Cloud GPU

1. Open your local application:
   ```powershell
   python video_dubbing_gui.py
   ```
2. Under **Execution Engine**, choose:
   - `☁️ Remote Cloud GPU (Google Colab / Kaggle)`
3. Paste the Cloudflare URL (e.g., `https://example-cloud-gpu.trycloudflare.com`).
4. Click **🔗 Test Connection** — it will confirm your connection to the cloud GPU (e.g., `✓ Connected: Tesla T4 (16.0 GB VRAM)`).
5. Select your video file or YouTube URL, configure your target language, and click **▶ Start Processing**.

### What Happens Automatically:
- Your local GUI uploads the video to the Cloud GPU.
- The Cloud GPU executes all stages (Whisper, Translation, XTTS voice cloning, Wav2Lip GAN).
- Real-time logs stream directly into your local GUI console.
- Once finished, the completed dubbed video is automatically downloaded to your local `results/` folder!
