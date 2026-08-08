#!/usr/bin/env python3
"""
=============================================================================
 AI Video Dubbing System - All-in-One Downloader & Environment Setup Script
=============================================================================
 This script automatically:
  1. Creates all required project folders.
  2. Downloads pre-trained AI model weights (Wav2Lip GAN & S3FD Face Detector)
     with real-time progress bars, speed, and resume support.
  3. Initializes NLTK tokenizers (punkt, punkt_tab).
  4. Generates .env configuration from .env.example if not present.
  5. Verifies FFmpeg, PyTorch, CUDA GPU acceleration, and core dependencies.
=============================================================================
"""

import os
import sys
import time
import shutil
import urllib.request
import argparse
from pathlib import Path

# Fix Windows console encoding for UTF-8 symbols
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure static_ffmpeg is active if installed
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except Exception:
    pass

# =============================================================================
# MODEL DEFINITIONS & TARGET PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODELS = [
    {
        "name": "Wav2Lip GAN Lip-Sync Weights",
        "filename": "wav2lip_gan.pth",
        "target_path": PROJECT_ROOT / "Wav2Lip" / "wav2lip_gan.pth",
        "alt_target": PROJECT_ROOT / "Wav2Lip" / "checkpoints" / "wav2lip_gan.pth",
        "urls": [
            "https://github.com/medahmedkrichen/ViDubb/releases/download/weights2/wav2lip_gan.1.1.pth",
            "https://huggingface.co/Akashk03/Wav2Lip/resolve/main/wav2lip_gan.pth"
        ],
        "expected_size": 435801865,  # ~415 MB
        "min_size_mb": 400
    },
    {
        "name": "S3FD Face Detection Model",
        "filename": "s3fd.pth",
        "target_path": PROJECT_ROOT / "Wav2Lip" / "face_detection" / "detection" / "sfd" / "s3fd.pth",
        "alt_target": None,
        "urls": [
            "https://github.com/medahmedkrichen/ViDubb/releases/download/weights1/s3fd-619a316812.1.1.pth",
            "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"
        ],
        "expected_size": 89843225,   # ~85.6 MB
        "min_size_mb": 80
    }
]

REQUIRED_DIRECTORIES = [
    PROJECT_ROOT / "Wav2Lip",
    PROJECT_ROOT / "Wav2Lip" / "checkpoints",
    PROJECT_ROOT / "Wav2Lip" / "face_detection" / "detection" / "sfd",
    PROJECT_ROOT / "Wav2Lip" / "speakers_image",
    PROJECT_ROOT / "Wav2Lip" / "results",
    PROJECT_ROOT / "Wav2Lip" / "temp",
    PROJECT_ROOT / "audio",
    PROJECT_ROOT / "audio_chunks",
    PROJECT_ROOT / "speakers_image",
    PROJECT_ROOT / "results",
    PROJECT_ROOT / "temp",
]

# =============================================================================
# DOWNLOAD HELPER WITH STREAMING PROGRESS BAR
# =============================================================================

def format_size(bytes_val: int) -> str:
    """Format bytes to human readable format (MB/GB)."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0 or unit == 'GB':
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} GB"

def download_file(urls: list, dest_path: Path, expected_size: int = None, min_size_mb: float = 0, name: str = "") -> bool:
    """Download a file with multi-URL fallback and streaming visual progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    # Check if already present and valid
    if dest_path.exists():
        actual_size = dest_path.stat().st_size
        if (expected_size and actual_size == expected_size) or (actual_size >= min_size_mb * 1024 * 1024):
            print(f"  [ALREADY PRESENT] {name or dest_path.name} ({format_size(actual_size)}) ✓")
            return True
        else:
            print(f"  [INCOMPLETE/INVALID] Found {dest_path.name} with size {format_size(actual_size)}, re-downloading...")
            try:
                dest_path.unlink()
            except Exception:
                pass

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    for idx, url in enumerate(urls):
        print(f"\n  [DOWNLOADING] {name or dest_path.name}")
        print(f"  Source: {url}")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                total_length = response.headers.get('content-length')
                if total_length:
                    total_bytes = int(total_length)
                else:
                    total_bytes = expected_size or 0

                downloaded = 0
                block_size = 1024 * 64  # 64 KB chunks
                start_time = time.time()

                with open(temp_path, 'wb') as out_file:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        downloaded += len(chunk)

                        # Progress calculation
                        elapsed = time.time() - start_time
                        speed = downloaded / elapsed if elapsed > 0 else 0
                        speed_str = f"{format_size(speed)}/s"

                        if total_bytes > 0:
                            percent = min(100.0, (downloaded / total_bytes) * 100)
                            bar_width = 30
                            filled = int(bar_width * (downloaded / total_bytes))
                            bar = "=" * filled + ">" if filled < bar_width else "=" * bar_width
                            bar = bar.ljust(bar_width)
                            print(f"\r  [{bar}] {percent:5.1f}% | {format_size(downloaded)} / {format_size(total_bytes)} | {speed_str}  ", end='', flush=True)
                        else:
                            print(f"\r  Downloaded: {format_size(downloaded)} | {speed_str}  ", end='', flush=True)

                print()

            # Finalize file
            if temp_path.exists():
                final_size = temp_path.stat().st_size
                if min_size_mb > 0 and final_size < (min_size_mb * 1024 * 1024):
                    print(f"  ✗ Downloaded file is too small ({format_size(final_size)}). Trying mirror...")
                    temp_path.unlink()
                    continue

                if dest_path.exists():
                    dest_path.unlink()
                temp_path.rename(dest_path)
                print(f"  ✓ Successfully saved: {dest_path.name} ({format_size(final_size)})")
                return True

        except Exception as e:
            print(f"\n  ✗ Download error from mirror #{idx+1}: {e}")
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

    print(f"  ✗ Failed to download {name} from all available sources.")
    return False

# =============================================================================
# SETUP STEPS
# =============================================================================

def step_create_directories():
    """Create all required directory structures."""
    print("\n" + "=" * 70)
    print("📁 STEP 1: Creating Directory Structure")
    print("=" * 70)
    for directory in REQUIRED_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)
        rel = directory.relative_to(PROJECT_ROOT)
        print(f"  ✓ {rel}/")
    print("  All directory structures ready!")

def step_download_models():
    """Download pre-trained Wav2Lip and face detection models."""
    print("\n" + "=" * 70)
    print("🧠 STEP 2: Downloading Pre-Trained AI Model Checkpoints")
    print("=" * 70)
    success_all = True
    for model_cfg in MODELS:
        ok = download_file(
            urls=model_cfg["urls"],
            dest_path=model_cfg["target_path"],
            expected_size=model_cfg.get("expected_size"),
            min_size_mb=model_cfg.get("min_size_mb", 0),
            name=model_cfg["name"]
        )
        if ok and model_cfg.get("alt_target"):
            alt_path = model_cfg["alt_target"]
            alt_path.parent.mkdir(parents=True, exist_ok=True)
            if not alt_path.exists():
                try:
                    shutil.copy2(model_cfg["target_path"], alt_path)
                    print(f"  ↳ Synced backup to: {alt_path.relative_to(PROJECT_ROOT)}")
                except Exception:
                    pass
        if not ok:
            success_all = False
    return success_all

def step_setup_env():
    """Ensure .env file exists with proper template."""
    print("\n" + "=" * 70)
    print("⚙️  STEP 3: Environment Configuration (.env)")
    print("=" * 70)
    env_file = PROJECT_ROOT / ".env"
    env_example = PROJECT_ROOT / ".env.example"

    if env_file.exists():
        print(f"  ✓ Existing .env file found at {env_file.name}")
    elif env_example.exists():
        shutil.copy2(env_example, env_file)
        print(f"  ✓ Created .env file from template .env.example")
    else:
        # Create fresh .env
        content = (
            "# Required for PyAnnote Speaker Diarization:\n"
            "HF_TOKEN=your_hugging_face_token_here\n\n"
            "# Optional for Groq Contextual LLM Translation:\n"
            "Groq_TOKEN=your_groq_api_key_here\n\n"
            "WHISPER_MODEL=large-v3\n"
            "DEFAULT_SOURCE_LANG=en\n"
            "DEFAULT_TARGET_LANG=es\n"
        )
        env_file.write_text(content, encoding='utf-8')
        print(f"  ✓ Generated default .env file")

    # Read environment status
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
    except Exception:
        pass

    hf_token = os.getenv("HF_TOKEN")
    groq_token = os.getenv("Groq_TOKEN")

    if not hf_token or "your_hugging_face_token" in hf_token:
        print("  ⚠️  HF_TOKEN is not set yet in .env!")
        print("     Get a free token at https://huggingface.co/settings/tokens")
        print("     and accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1")
    else:
        print(f"  ✓ HF_TOKEN is configured: {hf_token[:6]}...{hf_token[-4:]}")

    if not groq_token or "your_groq_token" in groq_token:
        print("  ℹ️  Groq_TOKEN is not set (optional - local MarianMT translation will be used).")
    else:
        print(f"  ✓ Groq_TOKEN is configured: {groq_token[:6]}...{groq_token[-4:]}")

def step_download_nltk():
    """Download required NLTK tokenizers."""
    print("\n" + "=" * 70)
    print("📚 STEP 4: Initializing NLTK Language Models")
    print("=" * 70)
    try:
        import nltk
        print("  Downloading punkt tokenizer...")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        print("  ✓ NLTK data initialized successfully!")
    except Exception as e:
        print(f"  ⚠️ NLTK download warning: {e}")

def step_verify_system():
    """Verify hardware, CUDA, PyTorch, and FFmpeg."""
    print("\n" + "=" * 70)
    print("🔍 STEP 5: System & Dependency Verification")
    print("=" * 70)

    # 1. Python
    print(f"  • Python Version : {sys.version.split()[0]} ({sys.platform})")

    # 2. PyTorch & CUDA
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "None (CPU Mode)"
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda_avail else 0
        print(f"  • PyTorch Version: {torch.__version__}")
        print(f"  • CUDA Enabled   : {'✓ YES' if cuda_avail else '✗ NO'}")
        print(f"  • GPU Device     : {gpu_name} ({vram_gb:.1f} GB VRAM)")
    except Exception as e:
        print(f"  • PyTorch        : ✗ Error ({e})")

    # 3. FFmpeg
    try:
        import subprocess
        res = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            first_line = res.stdout.splitlines()[0] if res.stdout else "Available"
            print(f"  • FFmpeg         : ✓ {first_line[:60]}")
        else:
            print("  • FFmpeg         : ⚠️ Process returned non-zero")
    except Exception:
        print("  • FFmpeg         : ⚠️ Command not in system PATH (static_ffmpeg fallback will be used)")

    # 4. Core Libraries Check
    check_libs = [
        ("pyannote.audio", "PyAnnote Speaker Diarization"),
        ("speechbrain", "SpeechBrain Voice Processing"),
        ("audio_separator", "Audio Separator MDX-Net"),
        ("faster_whisper", "Faster-Whisper Transcription"),
        ("transformers", "Transformers Translation"),
        ("pedalboard", "Pedalboard Audio Mastering"),
        ("pydub", "PyDub Audio Manipulation"),
        ("noisereduce", "Noise Reduction"),
        ("resemblyzer", "Resemblyzer Voice Profiling"),
        ("deepface", "DeepFace Face Matching"),
        ("gtts", "Google Text-to-Speech"),
        ("ffmpeg", "FFmpeg-Python"),
        ("yt_dlp", "YouTube Video Downloader"),
    ]

    print("\n  📦 Library Check:")
    for mod_name, label in check_libs:
        try:
            __import__(mod_name)
            print(f"    ✓ {label:<35} : OK")
        except ImportError:
            print(f"    ✗ {label:<35} : MISSING (run 'pip install {mod_name}')")

WHISPER_MODELS = {
    "tiny": ("Systran/faster-whisper-tiny", 75),
    "base": ("Systran/faster-whisper-base", 145),
    "small": ("Systran/faster-whisper-small", 480),
    "medium": ("Systran/faster-whisper-medium", 1530),
    "large-v3": ("Systran/faster-whisper-large-v3", 3100),
}

def step_download_whisper(model_names: list):
    """Pre-download faster-whisper models with clean progress indicators."""
    print("\n" + "=" * 70)
    print("🎙️  STEP 6: Pre-Downloading Whisper Speech-to-Text Models")
    print("=" * 70)
    try:
        from faster_whisper import WhisperModel
        for m in model_names:
            m_clean = m.lower().strip()
            size_mb = WHISPER_MODELS.get(m_clean, ("", 0))[1]
            size_str = f" (~{size_mb} MB)" if size_mb else ""
            print(f"\n  Downloading Faster-Whisper model: '{m_clean}'{size_str}...")
            print(f"  (Streaming from Hugging Face Hub, please wait)...")
            start_t = time.time()
            try:
                wm = WhisperModel(m_clean, device="cpu", compute_type="int8")
                elapsed = time.time() - start_t
                print(f"  ✓ Whisper model '{m_clean}' is ready! ({elapsed:.1f}s)")
                del wm
            except Exception as w_err:
                print(f"  ⚠️ Warning downloading '{m_clean}': {w_err}")
    except Exception as e:
        print(f"  ⚠️ Whisper download skipped: {e}")

def main():
    parser = argparse.ArgumentParser(description="AI Video Dubbing - All-in-One Downloader & Environment Setup")
    parser.add_argument("--models-only", action="store_true", help="Download only model weights")
    parser.add_argument("--verify-only", action="store_true", help="Verify current setup without downloading")
    parser.add_argument("--whisper", type=str, default="small", choices=['tiny', 'base', 'small', 'medium', 'large-v3', 'all', 'none'], help="Whisper model to pre-download (default: small)")
    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("🎬 AI VIDEO DUBBING SYSTEM - AUTOMATIC SETUP & DOWNLOADER")
    print("#" * 70)

    if args.verify_only:
        step_verify_system()
        return

    step_create_directories()
    step_setup_env()
    step_download_nltk()
    models_ok = step_download_models()

    if args.whisper != 'none':
        models_to_fetch = list(WHISPER_MODELS.keys()) if args.whisper == 'all' else [args.whisper]
        step_download_whisper(models_to_fetch)

    if not args.models_only:
        step_verify_system()

    print("\n" + "=" * 70)
    print("🎉 SETUP SUMMARY")
    print("=" * 70)
    if models_ok:
        print("  ✅ All model weights and required folders are ready!")
        print("  ✅ You can now launch the application:")
        print("\n     python video_dubbing_gui.py\n")
    else:
        print("  ⚠️ Some model weights failed to download automatically.")
        print("  Please check your internet connection and re-run:")
        print("     python download_and_setup.py")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
