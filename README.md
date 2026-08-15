# 🎬 AI Video Dubbing Studio 🤖✨

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20%7C%20CUDA%2012.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Flask](https://img.shields.io/badge/Flask-Web%20GUI-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-GPU%20Ready-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

**AI Video Dubbing Studio** is an advanced, production-ready AI video translation and dubbing pipeline. It automatically translates spoken video into any target language with **Zero-Shot Voice Cloning**, **Speaker Diarization**, **Background Music & Sound Effects Preservation (UVR)**, and **Wav2Lip GAN Lip-Synchronization**.

---

## 📑 Table of Contents

- [🌟 Features & Capabilities](#-features--capabilities)
- [🔄 The 9-Stage Dubbing Pipeline](#-the-9-stage-dubbing-pipeline)
- [🚀 3 Ways to Run](#-3-ways-to-run)
  - [Option 1: One-Click Google Colab (Easiest - No GPU Needed)](#option-1-one-click-google-colab-easiest---no-gpu-needed)
  - [Option 2: Hybrid Local GUI + Remote Cloud GPU (Recommended for Power Users)](#option-2-hybrid-local-gui--remote-cloud-gpu)
  - [Option 3: Full Local Installation (Run on Your PC)](#option-3-full-local-installation-run-on-your-pc)
- [📋 Prerequisites & System Requirements](#-prerequisites--system-requirements)
- [🛠️ Detailed Local Installation Guide](#️-detailed-local-installation-guide)
- [🔑 API Keys & Environment Configuration](#-api-keys--environment-configuration)
- [🎯 Running the Application](#-running-the-application)
  - [Web GUI (Modern Browser Interface)](#1-web-gui-recommended)
  - [Desktop GUI (Native Tkinter Window)](#2-desktop-gui-legacy)
  - [Remote Cloud GPU Server](#3-remote-cloud-gpu-server-colab--kaggle)
- [🌐 Supported Languages](#-supported-languages)
- [📁 Project Structure](#-project-structure)
- [🔧 Troubleshooting & FAQ](#-troubleshooting--faq)
- [🙏 Acknowledgments & Credits](#-acknowledgments--credits)
- [📝 License](#-license)

---

## 🌟 Features & Capabilities

- 🗣️ **Zero-Shot Multi-Speaker Voice Cloning**:
  - **Chatterbox Multilingual**: Ultra-realistic multilingual zero-shot voice cloning preserving tone, emotion, and accent.
  - **Coqui XTTS v2 & gTTS Fallbacks**: High-reliability automated fallback ensuring continuous synthesis.
- 👥 **PyAnnote Speaker Diarization & Gender Detection**:
  - Distinguishes individual speakers, speech turns, and estimates vocal pitch/formants to match appropriate voice characteristics.
- ⚡ **Faster-Whisper Transcription**:
  - State-of-the-art transcription speed and accuracy (supporting `tiny`, `base`, `small`, `medium`, and `large-v3` models).
- 🌍 **Context-Aware AI Translation & Duration Matching**:
  - Multi-tier translation chain with automatic fail-over: your own **Google Antigravity (`agy`) CLI** (optional, tried first if configured) → **Groq** → **Cerebras** → local **MarianMT**, with automated sentence condensing/expansion to fit original speech timing.
  - Multiple Groq/Cerebras API keys can be configured (1 required + up to 4 fallbacks each) — if one hits a rate limit mid-video, processing automatically shifts to the next.
- 📥 **Flexible Video Input**:
  - Upload a local file, paste a **YouTube URL**, or paste a **Google Drive share link** — all three work in the Web GUI, the Colab/Kaggle notebook, and the Remote Cloud GPU server.
- 🎵 **UVR Background Audio & Music Isolation**:
  - Separates vocals from background audio using **Ultimate Vocal Remover (MDX-Net)**. The original background music, sound effects, and ambient sounds are preserved and remixed behind the newly dubbed voices.
- 👄 **Wav2Lip GAN Lip-Synchronization**:
  - Frame-by-frame neural face rendering that realistically adjusts lip movements to match the new language audio.
- ☁️ **Cloud GPU Acceleration via Cloudflare Tunneling**:
  - Offload heavy computing to free **NVIDIA Tesla T4 / A100 / P100 GPUs** on Google Colab or Kaggle with zero local hardware requirements.
- 💻 **Dual User Interfaces**:
  - **Modern Web GUI** with real-time logs, video preview, audio playback, and progress indicators.
  - **Native Desktop GUI** for standalone desktop workflows.

---

## 🔄 The 9-Stage Dubbing Pipeline

```
  ┌─────────────────┐
  │  Input Video /  │
  │   YouTube URL   │
  └────────┬────────┘
           ▼
[Stage 1: Extract Audio] ───────────────► 16kHz Mono HQ WAV
           ▼
[Stage 2: Speaker Diarization] ─────────► PyAnnote + Gender/Pitch Analysis
           ▼
[Stage 3: Transcription] ───────────────► Faster-Whisper (Timestamps & Segments)
           ▼
[Stage 4: Translation & Condensing] ────► Antigravity / Groq / Cerebras / MarianMT
           ▼
[Stage 5: Voice Cloning (TTS)] ─────────► Chatterbox Multilingual / XTTS v2
           ▼
[Stage 6: Precise Audio Assembly] ──────► Time-stretched & Gap-Aligned Assembly
           ▼
[Stage 7: Background Music Isolation] ──► UVR MDX-Net Vocal / Music Split & Remix
           ▼
[Stage 8: Frame-Synced Video] ──────────► FFmpeg Synchronized Video Track
           ▼
[Stage 9: Wav2Lip GAN Lip-Sync] ────────► Neural Lip-Synced Video Render
           ▼
  ┌─────────────────┐
  │  Dubbed Output  │
  │    (MP4 Video)  │
  └─────────────────┘
```

1. **Stage 1 (Audio Extraction)**: Extracts high-fidelity 16kHz mono audio from video containers.
2. **Stage 2 (Speaker Diarization)**: Detects distinct speakers, speech intervals, and pitch characteristics.
3. **Stage 3 (Transcription)**: Generates timestamped word- and sentence-level transcripts using Faster-Whisper.
4. **Stage 4 (Translation & Condensing)**: Translates text (Antigravity → Groq → Cerebras → MarianMT fail-over chain) while algorithmically condensing phrasing to fit original timing windows.
5. **Stage 5 (Voice Synthesis)**: Clones the speaker's original voice into the target language.
6. **Stage 6 (Precise Assembly)**: Adjusts speech tempo with pitch-preserving time stretching to achieve perfect timeline sync.
7. **Stage 7 (Background Preservation)**: Uses UVR (MDX-Net) to separate original vocals and mix the instrumental track with dubbed speech.
8. **Stage 8 (Video Creation)**: Merges the newly assembled audio with the original video stream.
9. **Stage 9 (Lip-Syncing)**: Applies Wav2Lip GAN to re-animate mouth movements to match the new audio track.

---

## 🚀 3 Ways to Run

> **Kaggle users / want a 2nd GPU:** use [`colab_gpu_server_v2.ipynb`](colab_gpu_server_v2.ipynb) instead - it auto-detects Colab vs Kaggle, and on Kaggle's **GPU T4 x2** accelerator it automatically splits voice synthesis across both T4s in parallel. It's also crash-resumable: re-running its one-click cell after a kill/OOM picks up from the last completed stage/segment instead of starting over. Everything below still applies to the original `colab_gpu_server.ipynb` (Colab-only, single GPU).

### Option 1: One-Click Google Colab (Easiest - No GPU Needed)
*Best if you don't have an NVIDIA GPU or don't want to install anything locally.*

1. Open [`colab_gpu_server.ipynb`](colab_gpu_server.ipynb) in [Google Colab](https://colab.research.google.com/).
2. In the top menu, select **Runtime > Change runtime type** and choose **T4 GPU** (or A100).
3. Run **Cell 1** (Installs all dependencies and model weights).
4. Run **Cell 3** (One-Click Colab Form):
   - Choose **Upload from computer**, paste a **YouTube URL**, or paste a **Google Drive Link** (file must be shared as "Anyone with the link").
   - Select your **Target Language**, **Voice Quality**, and options.
   - Paste your **Groq API Key** (required for AI translation) and, optionally, up to **4 fallback keys** — if one hits Groq's rate limit mid-video, processing automatically shifts to the next key.
   - Optionally paste a **Cerebras API Key** too — a second AI-translation tier, tried automatically if every Groq key is unavailable.
   - Click **Run (▶)** — Colab will process the video, preview it in the notebook, and download it automatically.

---

### Option 2: Hybrid Local GUI + Remote Cloud GPU
*Best experience: run the responsive GUI on your PC, while a free Colab/Kaggle GPU handles heavy AI rendering.*

```
 ┌──────────────────────────┐          Secure Cloudflare Tunnel          ┌──────────────────────────┐
 │  Local PC (Web/App GUI)  │ ◄────────────────────────────────────────► │  Google Colab / Kaggle   │
 │  - Select Video          │        https://xxxx.trycloudflare.com       │  - Tesla T4 / A100 GPU   │
 │  - Control Settings      │                                            │  - Whisper + Wav2Lip     │
 │  - Live Progress Log     │                                            │  - Chatterbox + UVR      │
 └──────────────────────────┘                                            └──────────────────────────┘
```

1. **Start the Cloud GPU Server**:
   - Open [`colab_gpu_server.ipynb`](colab_gpu_server.ipynb) on Google Colab with a GPU runtime.
   - Run **Cell 1** (Setup).
   - Run **Cell 2** (Launch Server & Tunnel).
   - Copy the public Cloudflare URL output (e.g., `https://xxxx.trycloudflare.com`).
2. **Connect from Local GUI**:
   - Start your local GUI:
     ```bash
     python web_gui.py
     ```
   - In the **Execution Engine** dropdown, select **☁️ Remote Cloud GPU**.
   - Paste your Cloudflare URL and click **Test Connection**.
   - Video source can be a local upload, a **YouTube URL**, or a **Google Drive link**.
   - Process any video — it uploads to Colab, streams logs back to your browser, and saves the finished video to your local `results/` folder!

> **Optional — translate with your own [Google Antigravity](https://antigravity.google/) CLI instead of Groq/Cerebras:** run `antigravity_bridge/start_bridge.bat` on this same PC (agy must already be installed and logged in). It prints a free Cloudflare tunnel URL + a token — paste both into the **Antigravity Bridge URL / Bridge Token** fields shown under **☁️ Remote Cloud GPU**. The Kaggle/Colab side then reaches back into your own `agy` CLI for translation, ahead of Groq/Cerebras, before falling back to them automatically if the bridge is offline. Both change every time the bridge restarts, so re-paste them each session. See [`antigravity_bridge/`](antigravity_bridge/).

---

### Option 3: Full Local Installation (Run on Your PC)
*Best for offline use or if you have a local NVIDIA GPU (RTX 3060/4060 or higher recommended).*

1. Follow the [Detailed Local Installation Guide](#️-detailed-local-installation-guide) below.
2. Launch the [Web GUI](#1-web-gui-recommended) or the [Desktop GUI](#2-desktop-gui-legacy), and pick **💻 Local Machine** as the Execution Engine.
3. If the [Google Antigravity](https://antigravity.google/) `agy` CLI is installed on this PC, it's **detected and used automatically** for translation (ahead of Groq/Cerebras) — no setup, no URL, nothing to paste. Check the **System** panel in the Web GUI to confirm it was found.

---

## 📋 Prerequisites & System Requirements

| Component | Minimum | Recommended |
| :--- | :--- | :--- |
| **OS** | Windows 10/11, Ubuntu 20.04+, macOS | Windows 11 / Ubuntu 22.04 |
| **Python** | Python 3.10.x – 3.12.x | Python 3.10.11 or 3.11 |
| **RAM** | 8 GB | 16 GB+ |
| **GPU (Local)** | CPU Mode (slow) / GTX 1660 (6GB) | NVIDIA RTX 3060+ (8GB+ VRAM) |
| **FFmpeg** | Required in system PATH | Latest Static Build |
| **Build Tools** | Visual Studio C++ Build Tools (Windows) | VS 2022 C++ Tools / `build-essential` |

---

## 🛠️ Detailed Local Installation Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Asadullah404/Ai_Video_Dubbing.git
cd Ai_Video_Dubbing
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install PyTorch with CUDA Support
*If using an NVIDIA GPU, install PyTorch with matching CUDA drivers first:*
```bash
# For CUDA 12.1 / 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU Only
pip install torch torchvision torchaudio
```

### 4. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 5. Install Voice Cloning Engine & Setup Models
Run the automated downloader and environment setup script:
```bash
python download_and_setup.py
```
*Or manually download the pre-trained Wav2Lip and S3FD weights:*
```bash
# Linux / macOS / Git Bash
mkdir -p Wav2Lip/face_detection/detection/sfd
wget -c 'https://github.com/medahmedkrichen/ViDubb/releases/download/weights2/wav2lip_gan.1.1.pth' -O 'Wav2Lip/wav2lip_gan.pth'
wget -c 'https://github.com/medahmedkrichen/ViDubb/releases/download/weights1/s3fd-619a316812.1.1.pth' -O 'Wav2Lip/face_detection/detection/sfd/s3fd.pth'
```

### 6. Install FFmpeg
- **Windows**:
  1. Download the latest `ffmpeg-release-essentials.zip` from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) or [ffmpeg.org](https://ffmpeg.org/download.html).
  2. Extract to `C:\ffmpeg` and add `C:\ffmpeg\bin` to your System Environment Variables `PATH`.
  3. Verify in a new terminal: `ffmpeg -version`.
- **Ubuntu/Debian**:
  ```bash
  sudo apt update && sudo apt install -y ffmpeg
  ```
- **macOS**:
  ```bash
  brew install ffmpeg
  ```

---

## 🔑 API Keys & Environment Configuration

Create a `.env` file in the root directory (or copy `.env.example`):

```env
# Required for PyAnnote Speaker Diarization
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Required for AI-powered context-aware translation via Groq LLaMA-3 (Free at groq.com)
GROQ_TOKEN=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Optional: up to 4 fallback Groq keys - automatically used in order if the key above
# (or the one currently active) hits Groq's rate limit mid-video
GROQ_TOKEN_2=
GROQ_TOKEN_3=
GROQ_TOKEN_4=
GROQ_TOKEN_5=

# Which Groq model to use for translation - selectable per-run in the Web GUI / notebook too
GROQ_MODEL=llama-3.1-8b-instant

# Optional second AI-translation tier, tried automatically whenever Groq is unavailable
# (rate-limited/revoked key). Free at cloud.cerebras.ai - 1 required + up to 4 fallback keys.
CEREBRAS_API_KEY=
CEREBRAS_API_KEY_2=
CEREBRAS_API_KEY_3=
CEREBRAS_API_KEY_4=
CEREBRAS_API_KEY_5=
CEREBRAS_MODEL=llama-3.3-70b

# Optional - only for the Remote Cloud GPU flow (Cell 2 / colab_server.py): translate via YOUR
# OWN PC's Google Antigravity (`agy`) CLI instead of Groq/Cerebras, tried first. Run
# antigravity_bridge/start_bridge.bat on that PC - it prints these two values each session.
# Not needed for Local execution mode, which auto-detects `agy` with zero configuration.
ANTIGRAVITY_BRIDGE_URL=
ANTIGRAVITY_BRIDGE_TOKEN=

# Automatically accept Coqui TTS terms of service
COQUI_TOS_AGREED=1

# Optional: Path to exported YouTube cookies for restricted downloads
YT_COOKIES_FILE=
```

### How to obtain API tokens:
1. **Hugging Face Token (`HF_TOKEN`)**:
   - Sign up at [huggingface.co](https://huggingface.co).
   - Go to **Settings > Access Tokens** and create a read token.
   - Accept user conditions on the [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) model pages.
2. **Groq Token (`GROQ_TOKEN`)** *(Required for AI translation; if omitted, the next tier in the chain is used instead)*:
   - Create a free account at [console.groq.com](https://console.groq.com) and generate an API key.
   - Optionally create a few more free accounts and add their keys as `GROQ_TOKEN_2` .. `GROQ_TOKEN_5` — if the active key hits Groq's free-tier rate limit partway through a video, processing automatically shifts to the next one. The Web GUI and the Colab notebook both also let you type these 5 keys directly instead of using `.env`.
3. **Cerebras Token (`CEREBRAS_API_KEY`)** *(Optional fallback tier)*:
   - Create a free account at [cloud.cerebras.ai](https://cloud.cerebras.ai) and generate an API key (no credit card required) — its free tier (~1M tokens/day) is much larger than Groq's.
4. **Antigravity Bridge (`ANTIGRAVITY_BRIDGE_URL` / `_TOKEN`)** *(Optional, remote mode only)*:
   - Only relevant if you want the Kaggle/Colab side of a Remote Cloud GPU run to translate through your own [Google Antigravity](https://antigravity.google/) `agy` CLI instead of Groq/Cerebras. Run `antigravity_bridge/start_bridge.bat` on your PC and paste what it prints — see [`antigravity_bridge/`](antigravity_bridge/) for details. Local execution mode doesn't need this at all; it detects and uses `agy` automatically.

---

## 🎯 Running the Application

### 1. Web GUI (Recommended)
Launches the modern, responsive web application and opens it in your default browser at `http://127.0.0.1:5000`:
```bash
python web_gui.py
```
- Features drag-and-drop video upload, YouTube URL fetching, Google Drive link fetching, live progress logs, output video player, and one-click download.
- In **Local** execution mode, it auto-detects and uses your own [Google Antigravity](https://antigravity.google/) `agy` CLI for translation if installed — no configuration needed. In **Remote** mode, paste the Cloudflare tunnel URL/token from `antigravity_bridge/start_bridge.bat` to use it instead of Groq/Cerebras. Check the **System** panel for live status of all translation providers.

### 2. Desktop GUI (Legacy)
Launches the standalone Tkinter desktop interface:
```bash
python video_dubbing_gui.py
```

### 3. Remote Cloud GPU Server (Colab / Kaggle)
Starts the headless FastAPI server with Cloudflare tunneling:
```bash
python colab_server.py
```

---

## 🌐 Supported Languages

AI Video Dubbing Studio supports dozens of languages for speech recognition, translation, and neural voice synthesis:

| Code | Language | Code | Language | Code | Language |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `en` | English | `es` | Spanish | `fr` | French |
| `de` | German | `it` | Italian | `pt` | Portuguese |
| `pl` | Polish | `tr` | Turkish | `ru` | Russian |
| `nl` | Dutch | `cs` | Czech | `ar` | Arabic |
| `zh-cn` | Chinese | `ja` | Japanese | `ko` | Korean |
| `hi` | Hindi | `ur` | Urdu | `hu` | Hungarian |
| `bn` | Bengali | `ta` | Tamil | `te` | Telugu |
| `ml` | Malayalam | `th` | Thai | `vi` | Vietnamese |
| `id` | Indonesian | `ms` | Malay | `fa` | Persian |

---

## 📁 Project Structure

```
Ai_Video_Dubbing/
├── colab_gpu_server.ipynb    # 🚀 Google Colab GPU notebook (One-Click & Server, single GPU)
├── colab_gpu_server_v2.ipynb # 🚀 Colab + Kaggle notebook, dual-T4 sharded synthesis, crash-resumable
├── colab_server.py           # 🌐 Remote Cloud GPU FastAPI backend
├── remote_client.py          # 🔗 Cloudflare tunnel connector & task dispatcher
├── web_gui.py                # 💻 Flask Web GUI server
├── video_dubbing_gui.py      # 🖥️ Desktop (Tkinter) GUI
├── video_dubbing_core.py     # ⚙️ Full 9-stage core dubbing pipeline
├── antigravity_bridge/       # 🌉 Optional: translate via your own PC's Google Antigravity `agy` CLI
│   ├── bridge_server.py      #    HTTP bridge + Cloudflare tunnel (used by Remote mode)
│   ├── agy_headless.py       #    Reliable non-interactive `agy` calling (used locally too)
│   └── start_bridge.bat      #    One-click launcher (Windows)
├── download_and_setup.py     # 📦 Automated model downloader & setup helper
├── requirements.txt          # 📋 Project dependencies
├── COLAB_KAGGLE_GPU_GUIDE.md # 📖 Dedicated Cloud GPU setup manual
├── templates/                # 🎨 Web GUI HTML templates
├── static/                   # 🎨 Web GUI CSS, JavaScript, and assets
├── Wav2Lip/                  # 👄 Lip-synchronization GAN and face detector
│   ├── wav2lip_gan.pth
│   └── face_detection/
└── results/                  # 🎬 Output dubbed videos and audio
```

---

## 🔧 Troubleshooting & FAQ

### 1. YouTube Download Fails (Bot Check / 403)
Colab or cloud servers often encounter bot verification checks from YouTube.
- **Fix**: Use a browser extension (such as *Get cookies.txt LOCALLY*) to export cookies while logged into YouTube.
- Set `YT_COOKIES_FILE=/path/to/cookies.txt` or paste the path into the GUI / Colab form.

### 2. PyAnnote Diarization Error: `401 Unauthorized` or Model Access Restricted
- **Fix**: Ensure your `HF_TOKEN` is set in `.env` and you have clicked **"Agree and access repository"** on both [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).

### 3. Missing `ffmpeg` or `ffprobe`
- Ensure FFmpeg is installed and `ffmpeg -version` works in your command prompt / terminal. If you just added it to PATH, restart your terminal or IDE.

### 4. CUDA Out of Memory (OOM)
- If processing long videos on a GPU with limited VRAM (e.g. 6GB or 8GB), select Whisper model `small` or `base` and set Voice Quality to `high` or `standard`.
- Alternatively, use **Option 1 or Option 2** to run on a free 16GB Tesla T4 GPU on Google Colab.

### 5. Google Drive Download Fails
- Make sure the file's sharing setting is **"Anyone with the link"** — `gdown` can't fetch a private/restricted file without an interactive Google login, which none of these environments support.

### 6. Antigravity (`agy`) Translation Isn't Being Used
- **Local mode**: check the **System** panel in the Web GUI for the `agy` CLI status. If it says "Not found", `agy` isn't on this PC's `PATH` — install it, or just ignore it, since translation automatically falls back to Groq/Cerebras.
- **Remote mode**: make sure `antigravity_bridge/start_bridge.bat` is running on your PC and you pasted the *current* URL/token — both change every time the bridge restarts. Check that window's log for a Cloudflare tunnel URL.
- If the bridge/local call logs a model error, run `agy models` on that PC to see the exact current model slugs (Google renames/retires them periodically) and set the `AGY_MODEL` environment variable to whichever Flash-tier one is listed, before starting the bridge.
- Either way, this tier is entirely optional and fails safely — any problem with it just falls through to Groq → Cerebras → MarianMT automatically; it will never fail the whole job.

### 7. A Few Seconds of a Dubbed Segment Are Silent
- Every voice-cloning engine (Chatterbox, XTTS) and the gTTS fallback are retried automatically, but on rare occasions (e.g. a persistent network blip) a single segment can still fail everywhere. Rather than aborting the whole video, that one segment is left silent and logged with a warning. Re-running the same job (`--no_reset` on the CLI, or just starting it again in the GUIs) picks up and retries only the segments that are still missing, not the whole video.

---

## 🙏 Acknowledgments & Credits

This project builds upon pioneering open-source research and tools:
- **[ViDubb](https://github.com/medahmedkrichen/ViDubb)** by medahmedkrichen — Original foundation and architecture inspiration.
- **[Wav2Lip](https://github.com/Rudrabha/Wav2Lip)** (KR et al.) — Accurate speech-driven facial animation.
- **[Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)** (SYSTRAN) — High-efficiency Whisper implementation with CTranslate2.
- **[Chatterbox Multilingual](https://github.com/resemble-ai/chatterbox)** (Resemble AI) & **[Coqui TTS](https://github.com/idiap/coqui-ai-TTS)** — Multi-speaker Zero-Shot voice cloning.
- **[PyAnnote.audio](https://github.com/pyannote/pyannote-audio)** (Bredin et al.) — Neural speaker diarization.
- **[Audio Separator / UVR](https://github.com/nomadkaraoke/python-audio-separator)** — Ultimate Vocal Remover MDX-Net architecture.
- **[Groq](https://groq.com/)** & **[Cerebras](https://cerebras.ai/)** — high-speed LLM inference for context-aware translation.
- **[Google Antigravity](https://antigravity.google/)** — optional local/bridged translation tier via the `agy` CLI; the reliable non-interactive calling technique in `antigravity_bridge/agy_headless.py` is adapted from a [community fix](https://gist.github.com/allahsan/a9a9e9c8a49aecede67ce974e64ef3cf) for agy's Windows `--print` output bug.
- **[gdown](https://github.com/wkentaro/gdown)** — Google Drive video downloads.
- **[pycloudflared](https://github.com/Bing-su/pycloudflared) / Cloudflare Tunnel** — zero-config secure public tunneling for the Remote Cloud GPU server and the Antigravity bridge.

---

## 📝 License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for complete details.

---
*Developed with ❤️ for the global AI and content creator community.*
