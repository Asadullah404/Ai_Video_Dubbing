import os
import time
import requests
from pathlib import Path
from typing import Optional, Callable

class RemoteDubbingClient:
    """Client for offloading AI Video Dubbing to Google Colab or Kaggle GPU Server"""
    
    def __init__(self, server_url: str, log_callback: Optional[Callable[[str], None]] = None):
        self.server_url = server_url.rstrip("/")
        self.log = log_callback or print

    def check_health(self) -> dict:
        """Check if remote GPU server is online and retrieve hardware specs"""
        try:
            res = requests.get(f"{self.server_url}/health", timeout=10)
            if res.status_code == 200:
                return res.json()
            else:
                return {"status": "error", "message": f"HTTP {res.status_code}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def process_video(
        self,
        video_path: str,
        is_youtube: bool,
        source_lang: str,
        target_lang: str,
        whisper_model: str = "large-v3",
        voice_quality: str = "ultra",
        enable_lipsync: bool = True,
        preserve_bg: bool = True,
        hf_token: Optional[str] = None,
        groq_token: Optional[str] = None,
        groq_model: Optional[str] = None,
        cerebras_token: Optional[str] = None,
        cerebras_model: Optional[str] = None,
        output_dir: str = "results"
    ) -> Optional[str]:
        """Submit job to Colab GPU, stream logs live, and download output video"""
        self.log("\n" + "="*60)
        self.log(f"Connecting to Cloud GPU Server: {self.server_url}")
        self.log("="*60)
        
        health = self.check_health()
        if health.get("status") != "healthy":
            self.log(f"❌ Server connection failed: {health.get('message', 'Unknown error')}")
            raise Exception(f"Cannot connect to Cloud GPU server at {self.server_url}")
        
        gpu_info = health.get("gpu", {})
        self.log(f"✓ Cloud GPU Connected: {gpu_info.get('gpu_name', 'Unknown GPU')} ({gpu_info.get('vram_gb', 0)} GB VRAM)")
        
        # Prepare submission
        data = {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "whisper_model": whisper_model,
            "voice_quality": voice_quality,
            "enable_lipsync": str(enable_lipsync).lower(),
            "preserve_bg": str(preserve_bg).lower(),
        }
        if hf_token: data["hf_token"] = hf_token
        if groq_token: data["groq_token"] = groq_token
        if groq_model: data["groq_model"] = groq_model
        if cerebras_token: data["cerebras_token"] = cerebras_token
        if cerebras_model: data["cerebras_model"] = cerebras_model

        files = {}
        if is_youtube:
            data["youtube_url"] = video_path
            self.log(f"Submitting YouTube URL: {video_path}...")
            res = requests.post(f"{self.server_url}/dub", data=data, timeout=60)
        else:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            self.log(f"Uploading video file ({file_size_mb:.2f} MB) to Cloud GPU...")
            
            with open(video_path, "rb") as f:
                files = {"video_file": (Path(video_path).name, f, "video/mp4")}
                res = requests.post(f"{self.server_url}/dub", data=data, files=files, timeout=600)

        if res.status_code != 200:
            raise Exception(f"Job submission failed: {res.text}")
        
        task_info = res.json()
        task_id = task_info.get("task_id")
        self.log(f"✓ Job queued on Cloud GPU (Task ID: {task_id})")
        self.log("Streaming real-time execution logs from Cloud GPU...\n")
        
        # Poll status
        seen_logs_count = 0
        while True:
            time.sleep(2)
            try:
                status_res = requests.get(f"{self.server_url}/status/{task_id}", timeout=15)
                if status_res.status_code != 200:
                    continue
                
                status_data = status_res.json()
                status = status_data.get("status")
                logs = status_data.get("logs", [])
                
                # Print new logs
                if len(logs) > seen_logs_count:
                    new_logs = logs[seen_logs_count:]
                    for l in new_logs:
                        self.log(f"☁️ [Cloud GPU] {l}")
                    seen_logs_count = len(logs)
                
                if status == "completed":
                    self.log("\n✓ Remote GPU processing completed successfully!")
                    break
                elif status == "failed":
                    err = status_data.get("error", "Unknown error")
                    raise Exception(f"Remote GPU task failed: {err}")
                    
            except requests.RequestException as req_err:
                self.log(f"Network status check warning: {req_err}")
                time.sleep(2)

        # Download result
        os.makedirs(output_dir, exist_ok=True)
        local_output_path = os.path.join(output_dir, "dubbed_video.mp4")
        self.log(f"Downloading finished dubbed video to {local_output_path}...")
        
        with requests.get(f"{self.server_url}/download/{task_id}", stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(local_output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
        downloaded_mb = os.path.getsize(local_output_path) / (1024 * 1024)
        self.log(f"✓ Download complete! Saved to {local_output_path} ({downloaded_mb:.2f} MB)")
        return local_output_path
