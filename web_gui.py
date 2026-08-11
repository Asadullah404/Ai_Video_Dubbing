import os
import sys
import threading
import traceback
import warnings
import webbrowser
import importlib.util
from pathlib import Path

# Suppress non-critical library warnings (symlinks, TF32 reproducibility, future PyTorch weights_only)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings('ignore')

try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

# Compatibility patch for huggingface_hub (maps deprecated use_auth_token -> token)
try:
    import huggingface_hub
    _orig_hf_hub_download = huggingface_hub.hf_hub_download
    def _compat_hf_hub_download(*args, **kwargs):
        if 'use_auth_token' in kwargs:
            tok = kwargs.pop('use_auth_token')
            if 'token' not in kwargs and tok is not None:
                kwargs['token'] = tok
        return _orig_hf_hub_download(*args, **kwargs)
    huggingface_hub.hf_hub_download = _compat_hf_hub_download
except Exception:
    pass

try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except Exception:
    pass

import psutil
import torch
from dotenv import load_dotenv
from flask import Flask, jsonify, request, render_template, send_from_directory, abort
from werkzeug.utils import secure_filename

load_dotenv()

try:
    from video_dubbing_core import CONFIG, load_groq_keys_from_env
except ImportError:
    class CONFIG:
        chunk_duration = 300
    def load_groq_keys_from_env():
        token = os.getenv('Groq_TOKEN') or os.getenv('GROQ_TOKEN')
        return [token] if token else []

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}

ALL_LANGUAGES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
    'it': 'Italian', 'pt': 'Portuguese', 'pl': 'Polish', 'tr': 'Turkish',
    'ru': 'Russian', 'nl': 'Dutch', 'cs': 'Czech', 'ar': 'Arabic',
    'zh-cn': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean', 'hi': 'Hindi',
    'ur': 'Urdu', 'bn': 'Bengali', 'ta': 'Tamil', 'te': 'Telugu',
    'hu': 'Hungarian', 'th': 'Thai', 'vi': 'Vietnamese', 'id': 'Indonesian'
}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 * 1024  # 4 GB max upload

# ============================================================================
# JOB STATE (single-user local app - one job at a time, in-process)
# ============================================================================

job_lock = threading.Lock()
job_state = {
    'status': 'idle',       # idle | processing | completed | failed
    'logs': [],
    'error': None,
    'result_video': None,   # filename inside results/, if any
    'result_audio': None,
}


def reset_job_state():
    job_state['status'] = 'processing'
    job_state['logs'] = []
    job_state['error'] = None
    job_state['result_video'] = None
    job_state['result_audio'] = None


def job_log(message):
    with job_lock:
        job_state['logs'].append(str(message))


def run_dubbing_job(config: dict):
    """Runs in a background thread. Mirrors the exact same orchestration the old Tkinter
    GUI used (process_video) - only the UI layer changed, not the underlying dubbing logic
    in video_dubbing_core.py / remote_client.py, which is untouched."""
    try:
        job_log("=" * 60)

        if config['execution_mode'] == 'remote':
            job_log("Starting AI Video Dubbing on Cloud GPU (Google Colab / Kaggle)")
            job_log("=" * 60)
            job_log(f"Remote Server: {config['remote_url']}")
            job_log(f"Video Source: {config['video_path']}")
            job_log(f"Source: {config['source_lang']} -> Target: {config['target_lang']}")

            from remote_client import RemoteDubbingClient
            client = RemoteDubbingClient(config['remote_url'], log_callback=job_log)
            groq_keys = config['groq_keys'] if config['use_context'] else []
            if len(groq_keys) > 1:
                job_log(f"Groq: {len(groq_keys)} API keys configured (auto fail-over on rate limit)")
            output_file = client.process_video(
                video_path=config['video_path'],
                is_youtube=config['is_youtube'],
                source_lang=config['source_lang'],
                target_lang=config['target_lang'],
                whisper_model=config['whisper_model'],
                voice_quality=config['voice_quality'],
                enable_lipsync=config['enable_lipsync'],
                preserve_bg=config['preserve_bg'],
                hf_token=os.getenv('HF_TOKEN'),
                # The remote server's /dub endpoint takes a single string field - joined with
                # commas here, EnhancedVideoDubbing on the server splits it back into a key list.
                groq_token=','.join(groq_keys) if groq_keys else None,
                output_dir=str(RESULTS_DIR)
            )

            job_log("\n" + "=" * 60)
            job_log("CLOUD GPU PROCESSING COMPLETE!")
            job_log(f"Result saved locally: {output_file}")
            job_log("=" * 60)

            with job_lock:
                job_state['result_video'] = os.path.basename(output_file)
                job_state['status'] = 'completed'
            return

        # Local execution mode
        job_log("Starting Advanced Video Dubbing System (Local)")
        job_log("=" * 60)

        if config['is_youtube']:
            job_log("\nDownloading from YouTube...")
            from video_dubbing_core import download_youtube_video
            video_path = download_youtube_video(config['video_path'], job_log)
            if not video_path:
                job_log("ERROR: Download failed")
                with job_lock:
                    job_state['status'] = 'failed'
                    job_state['error'] = 'YouTube download failed'
                return
        else:
            video_path = config['video_path']

        job_log(f"\nVideo: {video_path}")
        job_log(f"Source: {config['source_lang']} -> Target: {config['target_lang']}")

        try:
            from video_dubbing_core import EnhancedVideoDubbing as VideoDubber
        except ImportError:
            from video_dubbing_core import UnlimitedVideoDubbing as VideoDubber

        groq_keys = config['groq_keys'] if config['use_context'] else []
        if len(groq_keys) > 1:
            job_log(f"Groq: {len(groq_keys)} API keys configured (auto fail-over on rate limit)")

        dubber = VideoDubber(
            video_path=video_path,
            source_lang=config['source_lang'],
            target_lang=config['target_lang'],
            whisper_model=config['whisper_model'],
            voice_quality=config['voice_quality'],
            enable_lipsync=config['enable_lipsync'],
            preserve_bg=config['preserve_bg'],
            hf_token=os.getenv('HF_TOKEN'),
            groq_token=groq_keys,
            log_callback=job_log
        )

        dubber.process()

        job_log("\n" + "=" * 60)
        job_log("PROCESSING COMPLETE!")
        job_log("=" * 60)

        result_video = RESULTS_DIR / "dubbed_video.mp4"
        result_audio = RESULTS_DIR / "dubbed_audio.mp3"
        with job_lock:
            job_state['result_video'] = result_video.name if result_video.exists() else None
            job_state['result_audio'] = result_audio.name if result_audio.exists() else None
            job_state['status'] = 'completed'

    except Exception as e:
        job_log(f"\nERROR: {str(e)}")
        job_log(traceback.format_exc())
        with job_lock:
            job_state['status'] = 'failed'
            job_state['error'] = str(e)


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    return render_template('index.html', languages=ALL_LANGUAGES)


@app.route('/api/system-info')
def system_info():
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    has_gpu = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_gpu else "None (CPU mode)"

    hf_token = bool(os.getenv('HF_TOKEN'))
    env_groq_keys = load_groq_keys_from_env()
    groq_token = bool(env_groq_keys)

    # Lightweight presence check (find_spec, not a real import) so this responds instantly -
    # Chatterbox/TTS packages take several seconds to actually import.
    if importlib.util.find_spec("chatterbox") is not None:
        engine = {'label': 'Chatterbox Multilingual', 'level': 'good'}
    elif importlib.util.find_spec("TTS") is not None:
        engine = {'label': 'XTTS v2 only (add chatterbox-tts)', 'level': 'warn'}
    else:
        engine = {'label': 'None found (gTTS only, no cloning)', 'level': 'bad'}

    return jsonify({
        'ram_gb': round(ram_gb, 1),
        'has_gpu': has_gpu,
        'gpu_name': gpu_name,
        'chunk_duration': CONFIG.chunk_duration,
        'hf_token': hf_token,
        'groq_token': groq_token,
        'groq_keys_count': len(env_groq_keys),
        'cloning_engine': engine,
    })


@app.route('/api/test-connection', methods=['POST'])
def test_connection():
    data = request.get_json(force=True) or {}
    url = (data.get('url') or '').strip()
    if not url:
        return jsonify({'status': 'error', 'message': 'No URL provided'}), 400

    try:
        from remote_client import RemoteDubbingClient
        client = RemoteDubbingClient(url)
        health = client.check_health()
        return jsonify(health)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file or not file.filename:
        return jsonify({'error': 'No file selected'}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({'error': f'Unsupported file type: {ext}'}), 400

    filename = secure_filename(file.filename)
    dest = UPLOAD_DIR / filename
    file.save(dest)

    return jsonify({'path': str(dest.resolve()), 'filename': filename})


@app.route('/api/start', methods=['POST'])
def start_job():
    with job_lock:
        if job_state['status'] == 'processing':
            return jsonify({'error': 'A job is already processing'}), 409

    data = request.get_json(force=True) or {}

    execution_mode = data.get('execution_mode', 'local')
    remote_url = (data.get('remote_url') or '').strip()
    video_source = data.get('video_source', 'local')
    video_path = (data.get('video_path') or '').strip()
    is_youtube = video_source == 'youtube'

    if execution_mode == 'remote' and not remote_url.startswith(('http://', 'https://')):
        return jsonify({'error': 'Please enter a valid Cloud GPU URL (e.g. https://xxxx.trycloudflare.com)'}), 400

    if is_youtube:
        if not video_path.startswith(('http://', 'https://')):
            return jsonify({'error': 'Please enter a valid YouTube URL'}), 400
    else:
        if not video_path or not os.path.exists(video_path):
            return jsonify({'error': 'Please upload a video file first'}), 400

    use_context = bool(data.get('use_context', True))

    # 1 required primary Groq key + up to 4 optional fallback keys, typed into the UI.
    # Falls back to the keys configured in .env (GROQ_TOKEN / GROQ_TOKEN_2..5) if the UI
    # fields were left blank.
    groq_keys = [k.strip() for k in (data.get('groq_api_keys') or []) if isinstance(k, str) and k.strip()]
    if use_context and not groq_keys:
        groq_keys = load_groq_keys_from_env()

    config = {
        'execution_mode': execution_mode,
        'remote_url': remote_url,
        'video_path': video_path,
        'is_youtube': is_youtube,
        'source_lang': data.get('source_lang', 'en'),
        'target_lang': data.get('target_lang', 'es'),
        'whisper_model': data.get('whisper_model', 'large-v3'),
        'voice_quality': data.get('voice_quality', 'ultra'),
        'enable_lipsync': bool(data.get('enable_lipsync', True)),
        'preserve_bg': bool(data.get('preserve_bg', True)),
        'use_context': use_context,
        'groq_keys': groq_keys,
    }

    reset_job_state()
    thread = threading.Thread(target=run_dubbing_job, args=(config,), daemon=True)
    thread.start()

    return jsonify({'status': 'started'})


@app.route('/api/logs')
def get_logs():
    since = request.args.get('since', 0, type=int)
    with job_lock:
        logs = job_state['logs'][since:]
        total = len(job_state['logs'])
        status = job_state['status']
        error = job_state['error']
        result_video = job_state['result_video']
        result_audio = job_state['result_audio']

    return jsonify({
        'logs': logs,
        'total': total,
        'status': status,
        'error': error,
        'result_video': result_video,
        'result_audio': result_audio,
    })


@app.route('/results/<path:filename>')
def serve_result(filename):
    safe_path = (RESULTS_DIR / filename).resolve()
    if RESULTS_DIR.resolve() not in safe_path.parents and safe_path != RESULTS_DIR.resolve():
        abort(403)
    if not safe_path.exists():
        abort(404)
    return send_from_directory(RESULTS_DIR, filename)


def main():
    port = 5000
    url = f"http://127.0.0.1:{port}"
    print("\n" + "=" * 60)
    print("AI Video Dubbing Studio - Web GUI")
    print("=" * 60)
    print(f"Opening {url} in your browser...")
    print("Press CTRL+C to stop the server.")
    print("=" * 60 + "\n")

    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host='127.0.0.1', port=port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
