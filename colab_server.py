import os
import sys
import shutil
import subprocess
import threading
import uuid
import time
import json
from pathlib import Path
from typing import Optional, Dict
import warnings

# Suppress library warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["COQUI_TOS_AGREED"] = "1"
warnings.filterwarnings('ignore')

try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

# Ensure all essential server & ML dependencies are present
REQUIRED_PACKAGES = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("python-multipart", "multipart"),
    ("pycloudflared", "pycloudflared"),
    ("nest-asyncio", "nest_asyncio"),
    ("resemblyzer", "resemblyzer"),
    ("coqui-tts", "TTS"),
    ("deep-translator", "deep_translator"),
    ("gTTS", "gtts"),
    ("pedalboard", "pedalboard"),
    ("noisereduce", "noisereduce"),
    ("faster-whisper", "faster_whisper"),
    ("audio-separator[gpu]", "audio_separator"),
]

for pkg_name, mod_name in REQUIRED_PACKAGES:
    try:
        __import__(mod_name)
    except ImportError:
        print(f"📦 Auto-installing missing dependency: {pkg_name}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg_name], check=False)
        except Exception as e:
            print(f"  Note: {pkg_name} install notice ({e})")

# Chatterbox Multilingual (primary zero-shot voice cloning engine, XTTS v2 above is the
# automatic fallback). Installed with --no-deps: its PyPI metadata hard-pins
# torch==2.6.0/transformers==5.2.0 and pulls in an unrelated gradio web UI, which would
# downgrade/replace the torch/transformers this GPU runtime already provides. It works
# fine against newer torch/transformers in practice.
try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # noqa: F401
except ImportError:
    print("📦 Auto-installing missing dependency: chatterbox-tts (--no-deps)...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q", "--no-deps",
            "chatterbox-tts", "diffusers==0.29.0", "resemble-perth", "conformer==0.3.2",
            "s3tokenizer", "pykakasi==2.3.0", "spacy-pkuseg", "pyloudnorm",
        ], check=False)
    except Exception as e:
        print(f"  Note: chatterbox-tts install notice ({e})")

import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI App
app = FastAPI(title="AI Video Dubbing Cloud GPU Server", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Workspace directories
BASE_DIR = Path("colab_workspace")
BASE_DIR.mkdir(parents=True, exist_ok=True)
TASKS_DIR = BASE_DIR / "tasks"
TASKS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory task tracking
TASKS: Dict[str, dict] = {}

def get_gpu_info() -> dict:
    has_gpu = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_gpu else "No GPU (CPU Mode)"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if has_gpu else 0.0
    return {
        "cuda_available": has_gpu,
        "gpu_name": gpu_name,
        "vram_gb": round(vram_gb, 2),
        "torch_version": torch.__version__,
        "device": "cuda" if has_gpu else "cpu"
    }

@app.get("/")
def home():
    gpu = get_gpu_info()
    return {
        "status": "online",
        "service": "AI Video Dubbing Cloud GPU Server",
        "gpu": gpu,
        "endpoints": {
            "health": "/health",
            "dub": "POST /dub",
            "status": "GET /status/{task_id}",
            "download": "GET /download/{task_id}"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "gpu": get_gpu_info(),
        "active_tasks": len([t for t in TASKS.values() if t.get("status") == "processing"])
    }

def run_dubbing_task(task_id: str, video_path: str, source_lang: str, target_lang: str,
                      whisper_model: str, voice_quality: str, enable_lipsync: bool,
                      preserve_bg: bool, hf_token: Optional[str], groq_token: Optional[str]):
    task_dir = TASKS_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    
    def log_cb(msg: str):
        print(f"[{task_id[:8]}] {msg}")
        if task_id in TASKS:
            TASKS[task_id]["logs"].append(msg)
            if "Stage 1" in msg: TASKS[task_id]["stage"] = "Stage 1: Extracting Audio"
            elif "Stage 2" in msg: TASKS[task_id]["stage"] = "Stage 2: Speaker Diarization"
            elif "Stage 3" in msg: TASKS[task_id]["stage"] = "Stage 3: Transcription"
            elif "Stage 4" in msg: TASKS[task_id]["stage"] = "Stage 4: Translation"
            elif "Stage 5" in msg: TASKS[task_id]["stage"] = "Stage 5: Voice Synthesis"
            elif "Stage 6" in msg: TASKS[task_id]["stage"] = "Stage 6: Audio Assembly"
            elif "Stage 7" in msg: TASKS[task_id]["stage"] = "Stage 7: Background Preservation"
            elif "Stage 8" in msg: TASKS[task_id]["stage"] = "Stage 8: Video Assembly"
            elif "Stage 9" in msg: TASKS[task_id]["stage"] = "Stage 9: Lip Synchronization"
            elif "COMPLETE" in msg: TASKS[task_id]["stage"] = "Completed"

    orig_cwd = os.getcwd()
    try:
        TASKS[task_id]["status"] = "processing"
        TASKS[task_id]["stage"] = "Initializing GPU Engine"
        
        os.chdir(str(task_dir))
        
        from video_dubbing_core import EnhancedVideoDubbing
        
        dubber = EnhancedVideoDubbing(
            video_path=video_path,
            source_lang=source_lang,
            target_lang=target_lang,
            whisper_model=whisper_model,
            voice_quality=voice_quality,
            enable_lipsync=enable_lipsync,
            preserve_bg=preserve_bg,
            hf_token=hf_token,
            groq_token=groq_token,
            log_callback=log_cb
        )
        
        dubber.process()
        
        output_video = task_dir / "results" / "dubbed_video.mp4"
        if not output_video.exists():
            output_video = Path("results/dubbed_video.mp4")
            
        if output_video.exists() and output_video.stat().st_size > 1000:
            TASKS[task_id]["status"] = "completed"
            TASKS[task_id]["stage"] = "Completed"
            TASKS[task_id]["output_path"] = str(output_video.absolute())
            log_cb(f"✓ Video successfully dubbed: {output_video.stat().st_size / (1024*1024):.2f} MB")
        else:
            raise Exception("Output video was not generated or is empty.")
            
    except Exception as e:
        import traceback
        err_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Error in task {task_id}: {err_msg}")
        TASKS[task_id]["status"] = "failed"
        TASKS[task_id]["error"] = str(e)
        TASKS[task_id]["logs"].append(f"ERROR: {str(e)}")
    finally:
        os.chdir(orig_cwd)

@app.post("/dub")
async def start_dubbing(
    background_tasks: BackgroundTasks,
    video_file: Optional[UploadFile] = File(None),
    youtube_url: Optional[str] = Form(None),
    source_lang: str = Form("en"),
    target_lang: str = Form("es"),
    whisper_model: str = Form("large-v3"),
    voice_quality: str = Form("ultra"),
    enable_lipsync: bool = Form(True),
    preserve_bg: bool = Form(True),
    hf_token: Optional[str] = Form(None),
    groq_token: Optional[str] = Form(None)
):
    task_id = str(uuid.uuid4())
    task_dir = TASKS_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Save input video
    if video_file:
        video_ext = Path(video_file.filename).suffix or ".mp4"
        local_video_path = str((task_dir / f"input_video{video_ext}").absolute())
        with open(local_video_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
    elif youtube_url:
        import yt_dlp
        local_video_path = str((task_dir / "input_video.mp4").absolute())
        ydl_opts = {
            'format': 'bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/bestvideo[ext=mp4]+bestaudio/best[ext=mp4]/best',
            'outtmpl': local_video_path,
            'merge_output_format': 'mp4',
            'quiet': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
    else:
        raise HTTPException(status_code=400, detail="Must provide either video_file or youtube_url")

    TASKS[task_id] = {
        "id": task_id,
        "status": "queued",
        "stage": "Queued",
        "created_at": time.time(),
        "source_lang": source_lang,
        "target_lang": target_lang,
        "logs": ["Task received and queued on Cloud GPU server."]
    }

    background_tasks.add_task(
        run_dubbing_task,
        task_id=task_id,
        video_path=local_video_path,
        source_lang=source_lang,
        target_lang=target_lang,
        whisper_model=whisper_model,
        voice_quality=voice_quality,
        enable_lipsync=enable_lipsync,
        preserve_bg=preserve_bg,
        hf_token=hf_token or os.getenv("HF_TOKEN"),
        groq_token=groq_token or os.getenv("Groq_TOKEN")
    )

    return {
        "task_id": task_id,
        "status": "queued",
        "message": "Video dubbing task scheduled on Cloud GPU server."
    }

@app.get("/status/{task_id}")
def check_status(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task ID not found")
    
    task = TASKS[task_id]
    return {
        "task_id": task_id,
        "status": task.get("status"),
        "stage": task.get("stage"),
        "logs": task.get("logs", [])[-25:],
        "error": task.get("error")
    }

@app.get("/download/{task_id}")
def download_result(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task ID not found")
    
    task = TASKS[task_id]
    if task.get("status") != "completed" or not task.get("output_path"):
        raise HTTPException(status_code=400, detail=f"Task is in state: {task.get('status')}")
    
    output_path = task["output_path"]
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found on disk")
        
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"dubbed_{task.get('target_lang', 'video')}.mp4"
    )

def start_server_with_tunnel(port: int = 8000):
    """Start uvicorn in a background thread and expose via Cloudflare Tunnel"""
    import threading
    import nest_asyncio
    nest_asyncio.apply()
    
    print("\n" + "="*60)
    print("🚀 Starting AI Video Dubbing GPU Server...")
    gpu = get_gpu_info()
    print(f"🎮 GPU Device: {gpu['gpu_name']} ({gpu['vram_gb']} GB VRAM)")
    print(f"⚡ CUDA Available: {gpu['cuda_available']}")
    print("="*60 + "\n")
    
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    time.sleep(2)
    
    try:
        from pycloudflared import try_cloudflare
        print("🌐 Creating secure public Cloudflare Tunnel...")
        tunnel_res = try_cloudflare(port=port)
        
        public_url = ""
        if hasattr(tunnel_res, 'tunnel_url'):
            public_url = str(tunnel_res.tunnel_url)
        elif hasattr(tunnel_res, 'url'):
            public_url = str(tunnel_res.url)
        elif hasattr(tunnel_res, 'urls'):
            urls = tunnel_res.urls
            public_url = str(urls[0]) if isinstance(urls, list) else str(urls)
        elif str(tunnel_res).startswith("http"):
            public_url = str(tunnel_res)
        else:
            import re
            m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", str(tunnel_res))
            if m:
                public_url = m.group(0)
                
        print("\n" + "*"*65)
        print("🎉 YOUR CLOUD GPU SERVER IS READY!")
        if public_url:
            print(f"🔗 Public URL: {public_url}")
        else:
            print(f"🔗 Tunnel Object: {tunnel_res}")
        print("👉 Copy and paste the https://...trycloudflare.com URL into your Local Dubbing GUI.")
        print("*"*65 + "\n")
    except Exception as e:
        print(f"Cloudflare tunnel notice: {e}")
        print(f"Server is running locally on http://127.0.0.1:{port}")

if __name__ == "__main__":
    start_server_with_tunnel(8000)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nServer shutting down.")
