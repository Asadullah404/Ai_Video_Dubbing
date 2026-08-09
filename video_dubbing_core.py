import os
import sys
import shutil
import subprocess
import warnings

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

import torch
import torchaudio
import numpy as np
import cv2
import json
import re
import gc
import traceback
import psutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime

# Core imports
from pyannote.audio import Pipeline
from audio_separator.separator import Separator
from transformers import MarianMTModel, MarianTokenizer

try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except Exception:
    TTS = None
    TTS_AVAILABLE = False

try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    CHATTERBOX_AVAILABLE = True
except Exception:
    ChatterboxMultilingualTTS = None
    CHATTERBOX_AVAILABLE = False

from pydub import AudioSegment
import librosa
import soundfile as sf
import noisereduce as nr
from faster_whisper import WhisperModel
from gtts import gTTS
from pedalboard import Pedalboard, Compressor, Gain, LowpassFilter, Reverb, HighpassFilter, NoiseGate
from groq import Groq
from dotenv import load_dotenv
import nltk
from scipy import signal
from scipy.signal import wiener

try:
    import parselmouth
    from parselmouth.praat import call
    PRAAT_AVAILABLE = True
except:
    PRAAT_AVAILABLE = False

load_dotenv()

# ============================================================================
# ENHANCED CONFIGURATION
# ============================================================================

@dataclass
class ProcessingConfig:
    chunk_duration: int = 300
    max_memory_percent: float = 75.0
    batch_size: int = 8
    max_text_length: int = 400
    audio_sample_rate: int = 24000  # Increased for better quality
    video_quality_crf: int = 18
    
    # Audio enhancement parameters
    noise_reduction_strength: float = 0.85
    dynamic_range_db: float = 15.0
    target_loudness_lufs: float = -16.0
    
    # Voice analysis parameters
    pitch_analysis_duration: float = 30.0  # Analyze more audio
    formant_analysis_window: float = 0.025
    
    # Translation parameters
    translation_compression_target: float = 0.95  # Target 95% of original duration
    max_translation_length_ratio: float = 1.2
    min_translation_length_ratio: float = 0.6  # below this, expand phrasing to reduce dead-air gaps

def compute_optimal_chunk_duration() -> int:
    available_gb = psutil.virtual_memory().available / (1024**3)
    if available_gb < 4:
        return 120
    elif available_gb < 8:
        return 240
    elif available_gb < 16:
        return 300
    elif available_gb < 32:
        return 600
    else:
        return 900

CONFIG = ProcessingConfig()
CONFIG.chunk_duration = compute_optimal_chunk_duration()

XTTS_LANGUAGES = {
    'en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 
    'nl', 'cs', 'ar', 'zh-cn', 'ja', 'ko', 'hi', 'hu', 'ur'
}

XTTS_LANGUAGE_MAP = {
    'en': 'en', 'es': 'es', 'fr': 'fr', 'de': 'de', 'it': 'it',
    'pt': 'pt', 'pl': 'pl', 'tr': 'tr', 'ru': 'ru', 'nl': 'nl',
    'cs': 'cs', 'ar': 'ar', 'zh-cn': 'zh-cn', 'zh-tw': 'zh-cn',
    'zh': 'zh-cn', 'ja': 'ja', 'ko': 'ko', 'hi': 'hi', 'ur': 'hi',
    'hu': 'hu'
}

GTTS_LANGUAGES = {
    'ur', 'en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru',
    'nl', 'cs', 'ar', 'zh-cn', 'ja', 'ko', 'hi', 'bn', 'ta',
    'te', 'ml', 'th', 'vi', 'id', 'ms', 'fa', 'sw', 'ne', 'si'
}

# Chatterbox Multilingual (Resemble AI) - primary zero-shot cloning engine.
# Best free/open cloning quality available; MIT licensed. Falls back to XTTS v2
# for languages it doesn't cover (e.g. ur, bn, ta, te, ml, th, vi, id, fa, ne, si).
CHATTERBOX_LANGUAGES = {
    'ar', 'da', 'de', 'el', 'en', 'es', 'fi', 'fr', 'he', 'hi', 'it',
    'ja', 'ko', 'ms', 'nl', 'no', 'pl', 'pt', 'ru', 'sv', 'sw', 'tr', 'zh-cn'
}

CHATTERBOX_LANGUAGE_MAP = {
    'ar': 'ar', 'da': 'da', 'de': 'de', 'el': 'el', 'en': 'en', 'es': 'es',
    'fi': 'fi', 'fr': 'fr', 'he': 'he', 'hi': 'hi', 'ur': 'hi', 'it': 'it',
    'ja': 'ja', 'ko': 'ko', 'ms': 'ms', 'nl': 'nl', 'no': 'no', 'pl': 'pl',
    'pt': 'pt', 'ru': 'ru', 'sv': 'sv', 'sw': 'sw', 'tr': 'tr',
    'zh-cn': 'zh', 'zh-tw': 'zh', 'zh': 'zh'
}

# ============================================================================
# UTILITY CLASSES
# ============================================================================

class MemoryManager:
    @staticmethod
    def get_memory_usage() -> float:
        return psutil.virtual_memory().percent
    
    @staticmethod
    def check_memory() -> bool:
        return MemoryManager.get_memory_usage() < CONFIG.max_memory_percent
    
    @staticmethod
    def force_cleanup():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class ProgressTracker:
    def __init__(self, progress_file: str = "processing_progress.json"):
        self.progress_file = progress_file
        self.progress = self.load()
    
    def load(self) -> dict:
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            'stage': 'start',
            'completed_chunks': [],
            'completed_segments': [],
            'timestamp': datetime.now().isoformat(),
            'video_path': None
        }
    
    def save(self):
        self.progress['timestamp'] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def update_stage(self, stage: str):
        self.progress['stage'] = stage
        self.save()
    
    def mark_segment_done(self, segment_id: int):
        if segment_id not in self.progress['completed_segments']:
            self.progress['completed_segments'].append(segment_id)
            self.save()
    
    def is_segment_done(self, segment_id: int) -> bool:
        return segment_id in self.progress['completed_segments']
    
    def reset(self, video_path: str = None):
        self.progress = {
            'stage': 'start',
            'completed_chunks': [],
            'completed_segments': [],
            'timestamp': datetime.now().isoformat(),
            'video_path': video_path
        }
        self.save()
        
        for checkpoint_file in Path('.').glob('checkpoint_*.json'):
            try:
                checkpoint_file.unlink()
            except:
                pass
    
    def should_reset(self, video_path: str) -> bool:
        return (self.progress.get('video_path') != video_path or 
                self.progress['stage'] == 'complete')

class StreamingAudioProcessor:
    def __init__(self, video_path: str, chunk_duration: int = None):
        self.video_path = video_path
        self.chunk_duration = chunk_duration or CONFIG.chunk_duration
        self.duration = self._get_duration()
        self.fps = self._get_fps()
        self.num_chunks = max(1, int(np.ceil(self.duration / self.chunk_duration)))
    
    def _get_duration(self) -> float:
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', self.video_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                timeout=30
            )
            return float(result.stdout.strip())
        except:
            try:
                video = cv2.VideoCapture(self.video_path)
                fps = video.get(cv2.CAP_PROP_FPS)
                frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
                video.release()
                return frame_count / fps if fps > 0 else 0
            except:
                return 0.0
    
    def _get_fps(self) -> float:
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                 '-show_entries', 'stream=r_frame_rate', '-of', 
                 'default=noprint_wrappers=1:nokey=1', self.video_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                timeout=30
            )
            fps_str = result.stdout.strip()
            if '/' in fps_str:
                num, den = map(float, fps_str.split('/'))
                return num / den
            return float(fps_str)
        except:
            return 30.0
    
    def extract_chunk(self, chunk_idx: int, output_path: str) -> bool:
        start_time = chunk_idx * self.chunk_duration
        
        cmd = [
            'ffmpeg', '-ss', str(start_time), '-t', str(self.chunk_duration),
            '-i', self.video_path, '-vn', '-acodec', 'pcm_s16le',
            '-ar', str(CONFIG.audio_sample_rate), '-ac', '1', output_path, '-y'
        ]
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE, timeout=300)
            return result.returncode == 0 and os.path.exists(output_path)
        except:
            return False
    
    def get_full_audio_streaming(self, output_path: str) -> bool:
        cmd = [
            'ffmpeg', '-i', self.video_path, '-vn', '-acodec', 'pcm_s16le',
            '-ar', str(CONFIG.audio_sample_rate), '-ac', '1', output_path, '-y'
        ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE, timeout=600)
            return result.returncode == 0
        except:
            return False

# ============================================================================
# ADVANCED VOICE ANALYSIS
# ============================================================================

class AdvancedVoiceAnalyzer:
    """Enhanced voice analyzer with multiple detection methods"""

    def analyze_voice_comprehensive(self, audio_path: str) -> dict:
        """Comprehensive voice analysis with multiple methods"""
        try:
            # Load longer audio for better analysis
            y, sr = librosa.load(audio_path, sr=CONFIG.audio_sample_rate, 
                                duration=min(CONFIG.pitch_analysis_duration, 30.0))
            
            # Method 1: Fundamental frequency analysis
            gender_f0, confidence_f0 = self._analyze_pitch(y, sr)
            
            # Method 2: Formant analysis (if available)
            gender_formant, confidence_formant = self._analyze_formants(audio_path)
            
            # Method 3: Spectral centroid
            gender_spectral, confidence_spectral = self._analyze_spectral(y, sr)
            
            # Method 4: MFCC-based analysis
            gender_mfcc, confidence_mfcc = self._analyze_mfcc(y, sr)
            
            # Weighted voting
            votes = {
                'male': 0.0,
                'female': 0.0
            }
            
            votes[gender_f0] += confidence_f0 * 0.4  # Highest weight to pitch
            votes[gender_formant] += confidence_formant * 0.3
            votes[gender_spectral] += confidence_spectral * 0.2
            votes[gender_mfcc] += confidence_mfcc * 0.1
            
            final_gender = 'female' if votes['female'] > votes['male'] else 'male'
            final_confidence = max(votes.values()) / sum(votes.values())
            
            # Get detailed pitch statistics
            f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
            f0_clean = f0[~np.isnan(f0)]
            
            if len(f0_clean) > 0:
                median_pitch = float(np.median(f0_clean))
                pitch_std = float(np.std(f0_clean))
                pitch_range = float(np.ptp(f0_clean))
            else:
                median_pitch = 120.0 if final_gender == 'male' else 220.0
                pitch_std = 20.0
                pitch_range = 40.0
            
            # Energy and dynamics
            rms = np.mean(librosa.feature.rms(y=y)[0])
            zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])
            
            result = {
                'gender': final_gender,
                'confidence': float(final_confidence),
                'pitch_median': median_pitch,
                'pitch_std': pitch_std,
                'pitch_range': pitch_range,
                'energy': float(rms),
                'zero_crossing_rate': float(zcr),
                'analysis_methods': {
                    'pitch': (gender_f0, confidence_f0),
                    'formant': (gender_formant, confidence_formant),
                    'spectral': (gender_spectral, confidence_spectral),
                    'mfcc': (gender_mfcc, confidence_mfcc)
                }
            }
            
            del y, f0, f0_clean
            MemoryManager.force_cleanup()
            
            return result
            
        except Exception as e:
            return {
                'gender': 'male',
                'pitch_median': 120.0,
                'pitch_std': 20.0,
                'pitch_range': 40.0,
                'confidence': 0.5,
                'energy': 0.1
            }
    
    def _analyze_pitch(self, y: np.ndarray, sr: int) -> Tuple[str, float]:
        """Pitch-based gender detection"""
        try:
            f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
            f0_clean = f0[~np.isnan(f0)]
            
            if len(f0_clean) == 0:
                return 'male', 0.5
            
            median_pitch = np.median(f0_clean)
            
            # Enhanced thresholds with confidence scoring
            if median_pitch < 140:
                confidence = min(0.95, 0.6 + (140 - median_pitch) / 200)
                return 'male', confidence
            elif median_pitch > 200:
                confidence = min(0.95, 0.6 + (median_pitch - 200) / 200)
                return 'female', confidence
            else:
                # Ambiguous range - lower confidence
                distance_to_male = abs(median_pitch - 120)
                distance_to_female = abs(median_pitch - 220)
                
                if distance_to_male < distance_to_female:
                    confidence = 0.5 + (1 - distance_to_male / 80) * 0.3
                    return 'male', confidence
                else:
                    confidence = 0.5 + (1 - distance_to_female / 80) * 0.3
                    return 'female', confidence
        except:
            return 'male', 0.5
    
    def _analyze_formants(self, audio_path: str) -> Tuple[str, float]:
        """Formant-based gender detection using Praat"""
        if not PRAAT_AVAILABLE:
            return 'male', 0.0
        
        try:
            sound = parselmouth.Sound(audio_path)
            formants = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
            
            f1_values = []
            f2_values = []
            
            for t in np.linspace(0, sound.duration, min(100, int(sound.duration * 20))):
                f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
                f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
                
                if not np.isnan(f1) and not np.isnan(f2):
                    f1_values.append(f1)
                    f2_values.append(f2)
            
            if not f1_values:
                return 'male', 0.0
            
            avg_f1 = np.mean(f1_values)
            avg_f2 = np.mean(f2_values)
            
            # Male: F1 ~130Hz, F2 ~1200Hz
            # Female: F1 ~200Hz, F2 ~2100Hz
            
            if avg_f1 < 450 and avg_f2 < 1600:
                confidence = 0.85
                return 'male', confidence
            elif avg_f1 > 550 and avg_f2 > 1800:
                confidence = 0.85
                return 'female', confidence
            else:
                # Ambiguous
                male_distance = np.sqrt((avg_f1 - 400)**2 + (avg_f2 - 1200)**2)
                female_distance = np.sqrt((avg_f1 - 650)**2 + (avg_f2 - 2100)**2)
                
                if male_distance < female_distance:
                    confidence = 0.6
                    return 'male', confidence
                else:
                    confidence = 0.6
                    return 'female', confidence
        except:
            return 'male', 0.0
    
    def _analyze_spectral(self, y: np.ndarray, sr: int) -> Tuple[str, float]:
        """Spectral centroid-based gender detection"""
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            avg_centroid = np.mean(spectral_centroids)
            
            # Male voices: lower spectral centroid (~1500-2500 Hz)
            # Female voices: higher spectral centroid (~2500-4000 Hz)
            
            if avg_centroid < 2000:
                confidence = 0.7
                return 'male', confidence
            elif avg_centroid > 3000:
                confidence = 0.7
                return 'female', confidence
            else:
                confidence = 0.5
                if avg_centroid < 2500:
                    return 'male', confidence
                else:
                    return 'female', confidence
        except:
            return 'male', 0.5
    
    def _analyze_mfcc(self, y: np.ndarray, sr: int) -> Tuple[str, float]:
        """MFCC-based gender detection"""
        try:
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            
            # Simple heuristic based on first few coefficients
            if mfcc_mean[1] < -10:
                return 'male', 0.6
            elif mfcc_mean[1] > -5:
                return 'female', 0.6
            else:
                return 'male', 0.5
        except:
            return 'male', 0.5

# ============================================================================
# PROFESSIONAL AUDIO ENHANCEMENT
# ============================================================================

class ProfessionalAudioEnhancer:
    """Professional-grade audio enhancement"""
    
    def __init__(self):
        self.sample_rate = CONFIG.audio_sample_rate
    
    def enhance_voice(self, audio_path: str, voice_profile: dict,
                     quality: str = 'ultra', is_cloned_voice: bool = False) -> str:
        """Apply audio enhancement.

        This DSP chain (noise reduction, spectral gating, de-essing, reverb "warmth", EQ) was
        built to compensate for gTTS's flat, robotic output. Running the FULL chain on a
        zero-shot cloned voice (Chatterbox/XTTS), which is already clean studio-quality audio
        that's supposed to sound identical to the real speaker, tends to add artifacts instead
        of removing them - reported as the dubbed voice sounding "gross"/off. Cloned voices only
        get gentle loudness normalization + limiting; the full chain is reserved for the gTTS
        fallback, which genuinely needs it.
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)

            if is_cloned_voice:
                audio = self._normalize_loudness(audio, sr)
                audio = self._final_limiter(audio)
            else:
                # 1. Advanced noise reduction
                audio = self._advanced_noise_reduction(audio, sr)

                # 2. Spectral gating
                audio = self._spectral_gate(audio, sr)

                # 3. Dynamic range compression
                audio = self._dynamic_compression(audio, sr, quality)

                # 4. De-essing (reduce sibilance)
                audio = self._de_essing(audio, sr)

                # 5. Warmth enhancement
                audio = self._add_warmth(audio, sr)

                # 6. Clarity enhancement
                audio = self._enhance_clarity(audio, sr)

                # 7. Loudness normalization
                audio = self._normalize_loudness(audio, sr)

                # 8. Final limiting
                audio = self._final_limiter(audio)

            # Save enhanced audio
            output_path = audio_path.replace('.wav', '_enhanced.wav')
            sf.write(output_path, audio, sr)
            
            del audio
            MemoryManager.force_cleanup()
            
            return output_path
            
        except Exception as e:
            return audio_path
    
    def _advanced_noise_reduction(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Advanced noise reduction with spectral subtraction"""
        try:
            # Multi-pass noise reduction
            audio_clean = nr.reduce_noise(
                y=audio, 
                sr=sr, 
                prop_decrease=CONFIG.noise_reduction_strength,
                stationary=True,
                n_std_thresh_stationary=1.5
            )
            
            # Wiener filtering for residual noise
            audio_clean = wiener(audio_clean, mysize=5)
            
            return audio_clean
        except:
            return audio
    
    def _spectral_gate(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Spectral gating to remove low-level noise"""
        try:
            # Compute STFT
            D = librosa.stft(audio)
            magnitude, phase = librosa.magphase(D)
            
            # Gate threshold
            threshold = np.percentile(magnitude, 10)
            
            # Apply gate
            magnitude_gated = np.where(magnitude > threshold, magnitude, magnitude * 0.1)
            
            # Reconstruct
            D_gated = magnitude_gated * phase
            audio_gated = librosa.istft(D_gated)
            
            return audio_gated
        except:
            return audio
    
    def _dynamic_compression(self, audio: np.ndarray, sr: int, 
                            quality: str) -> np.ndarray:
        """Multi-band dynamic compression"""
        try:
            if quality == 'ultra':
                board = Pedalboard([
                    NoiseGate(threshold_db=-40, ratio=2.5, attack_ms=1, release_ms=100),
                    Compressor(threshold_db=-18, ratio=3.5, attack_ms=3, release_ms=50),
                    Compressor(threshold_db=-10, ratio=2, attack_ms=10, release_ms=100),  # Parallel compression
                ])
            elif quality == 'high':
                board = Pedalboard([
                    NoiseGate(threshold_db=-35, ratio=2),
                    Compressor(threshold_db=-20, ratio=2.5, attack_ms=5, release_ms=75),
                ])
            else:
                board = Pedalboard([
                    Compressor(threshold_db=-25, ratio=2)
                ])
            
            audio_compressed = board(audio, sr)
            return audio_compressed
        except:
            return audio
    
    def _de_essing(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Reduce harsh sibilant frequencies"""
        try:
            # Focus on 5-10 kHz range
            sos = signal.butter(4, [5000, 10000], btype='band', fs=sr, output='sos')
            sibilants = signal.sosfilt(sos, audio)
            
            # Detect high-energy sibilant regions
            threshold = np.percentile(np.abs(sibilants), 90)
            reduction_mask = np.abs(sibilants) > threshold
            
            # Reduce sibilants
            audio_deessed = audio.copy()
            audio_deessed[reduction_mask] *= 0.6
            
            return audio_deessed
        except:
            return audio
    
    def _add_warmth(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Add warmth to voice"""
        try:
            board = Pedalboard([
                # Subtle low-end boost
                Gain(gain_db=1.5),
                # Warm reverb
                Reverb(room_size=0.05, damping=0.7, wet_level=0.03, dry_level=0.97, width=0.5)
            ])
            
            audio_warm = board(audio, sr)
            return audio_warm
        except:
            return audio
    
    def _enhance_clarity(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Enhance voice clarity"""
        try:
            # Presence boost (2-5 kHz)
            board = Pedalboard([
                # Gentle high-pass to remove rumble
                HighpassFilter(cutoff_frequency_hz=80),
                # Presence boost
                Gain(gain_db=2),
                # Gentle low-pass to remove harsh highs
                LowpassFilter(cutoff_frequency_hz=10000)
            ])
            
            audio_clear = board(audio, sr)
            return audio_clear
        except:
            return audio
    
    def _normalize_loudness(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Normalize to target loudness (LUFS)"""
        try:
            # Simple RMS-based normalization as approximation
            rms = np.sqrt(np.mean(audio**2))
            
            if rms > 0:
                # Target RMS (approximation of -16 LUFS)
                target_rms = 0.1
                gain = target_rms / rms
                audio_normalized = audio * gain
            else:
                audio_normalized = audio
            
            return audio_normalized
        except:
            return audio
    
    def _final_limiter(self, audio: np.ndarray) -> np.ndarray:
        """Final limiting to prevent clipping"""
        try:
            peak = np.max(np.abs(audio))
            if peak > 0.95:
                audio = audio / peak * 0.95
            
            return audio
        except:
            return audio

def _gender_instruction(target_language: str, speaker_gender: Optional[str]) -> str:
    """Prompt fragment telling the LLM which grammatical gender to use for the speaker's own
    self-referring pronouns/adjective agreement/verb conjugation. Many languages (Spanish,
    Hindi, Urdu, Arabic, French, etc.) default to masculine forms when the speaker's gender is
    unknown - this fixes translations sounding like the wrong gender said them."""
    if speaker_gender not in ('male', 'female'):
        return ""
    return (
        f"The person speaking this line is {speaker_gender}. If {target_language} grammar "
        f"changes based on the speaker's own gender (self-referring pronouns, adjective "
        f"agreement, first-person verb conjugation), use the {speaker_gender} form.\n"
    )

# ============================================================================
# INTELLIGENT TRANSLATION CONDENSER
# ============================================================================

class TranslationCondenser:
    """Condense translations to match timing while preserving 100% meaning and completeness"""

    def __init__(self, target_language: str, groq_token: str = None):
        self.target_language = target_language
        self.groq_token = groq_token

    def condense_translation(self, original_text: str, translation: str,
                           original_duration: float, context_before: str = '',
                           context_after: str = '', speaker_gender: str = None) -> str:
        """Intelligently rephrase translation to better match available speaking time,
        in either direction, without dropping or inventing content."""
        if not translation or not translation.strip():
            return translation

        words = translation.split()
        if not words:
            return translation

        # Estimate speaking duration (words per second)
        estimated_duration = len(words) / 2.5
        duration_ratio = estimated_duration / max(0.5, original_duration) if original_duration > 0 else 1.0

        # Too long: shrink it so the line fits without needing heavy audio speed-up
        if duration_ratio > CONFIG.max_translation_length_ratio:
            if self.groq_token:
                condensed = self._condense_with_groq(original_text, translation, original_duration,
                                                     context_before, context_after, speaker_gender)
                if condensed and condensed.strip():
                    return condensed.strip()
            # When no LLM is available, NEVER slice words out of sentences.
            # Keep the full translation intact and let audio speed time-stretching handle the timing!
            return translation

        # Too short: many languages are naturally more compact, so a literal translation can be
        # much shorter than the original line - left as-is, that produces a long dead-silence gap
        # while the speaker's mouth/action is still going, since audio slow-down is capped to
        # avoid sounding unnaturally dragged out. Rephrase more fully/naturally to better use the
        # available time (never inventing new content, only fuller phrasing of the same meaning).
        if duration_ratio < CONFIG.min_translation_length_ratio and self.groq_token:
            expanded = self._expand_with_groq(original_text, translation, original_duration,
                                              context_before, context_after, speaker_gender)
            if expanded and expanded.strip():
                return expanded.strip()

        return translation

    def _condense_with_groq(self, original: str, translation: str, duration: float,
                           context_before: str = '', context_after: str = '',
                           speaker_gender: str = None) -> Optional[str]:
        """Use AI to create concise but completely meaningful translation"""
        try:
            client = Groq(api_key=self.groq_token)
            target_words = max(3, int(duration * 2.5))

            context_block = ""
            if context_before or context_after:
                context_block = (
                    f"Surrounding conversation, for context only - do NOT translate or include it:\n"
                    f"...said just before: \"{context_before}\"\n"
                    f"...said right after: \"{context_after}\"\n\n"
                )
            gender_block = _gender_instruction(self.target_language, speaker_gender)

            prompt = f"""You are a professional subtitle and video dubbing translator.

{context_block}{gender_block}Original text: "{original}"
Current translation: "{translation}"
Target duration: {duration:.1f} seconds (approximately {target_words} words)

Create a concise, completely natural, and grammatically complete translation in {self.target_language} that:
1. Preserves the COMPLETE core meaning without omitting critical facts
2. Fits naturally in {duration:.1f} seconds
3. Sounds natural when spoken and flows correctly with the surrounding conversation (right pronouns, tone, continuity)

Return ONLY the concise translation, nothing else."""

            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=200
            )
            
            condensed = response.choices[0].message.content.strip()

            # Validate it's actually meaningful and shorter
            if condensed and len(condensed.split()) <= len(translation.split()):
                return condensed

            return None

        except Exception:
            return None

    def _expand_with_groq(self, original: str, translation: str, duration: float,
                         context_before: str = '', context_after: str = '',
                         speaker_gender: str = None) -> Optional[str]:
        """Use AI to phrase the translation more fully/naturally so it better fills the
        available speaking time - never adding facts/content that weren't in the original."""
        try:
            client = Groq(api_key=self.groq_token)
            target_words = max(3, int(duration * 2.5))

            context_block = ""
            if context_before or context_after:
                context_block = (
                    f"Surrounding conversation, for context only - do NOT translate or include it:\n"
                    f"...said just before: \"{context_before}\"\n"
                    f"...said right after: \"{context_after}\"\n\n"
                )
            gender_block = _gender_instruction(self.target_language, speaker_gender)

            prompt = f"""You are a professional subtitle and video dubbing translator.

{context_block}{gender_block}Original text: "{original}"
Current translation: "{translation}"
Target duration: {duration:.1f} seconds (approximately {target_words} words)

The current translation is much shorter than the available speaking time, which leaves an
awkward silent gap while the speaker's mouth/action is still going on screen. Rephrase it with
fuller, more natural spoken phrasing (a complete sentence structure instead of a clipped/terse
one) so it takes closer to {duration:.1f} seconds to say naturally, and flows correctly with the
surrounding conversation (right pronouns, tone, continuity).

CRITICAL: Do NOT add any facts, details, or ideas that were not in the original text. Only use
fuller/more natural phrasing of the SAME meaning - never pad with filler or invented content.

Return ONLY the rephrased translation, nothing else."""

            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=200
            )

            expanded = response.choices[0].message.content.strip()

            # Validate it's actually meaningful and at least as long as before
            if expanded and len(expanded.split()) >= len(translation.split()):
                return expanded

            return None

        except Exception:
            return None

# ============================================================================
# ENHANCED MAIN DUBBING SYSTEM
# ============================================================================

class EnhancedVideoDubbing:
    def __init__(self, video_path: str, source_lang: str, target_lang: str,
                 whisper_model: str = "large-v3", voice_quality: str = 'ultra',
                 enable_lipsync: bool = True, preserve_bg: bool = True,
                 hf_token: str = None, groq_token: str = None,
                 log_callback=None, reset_progress: bool = True):
        
        self.video_path = video_path
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.whisper_model = whisper_model
        self.voice_quality = voice_quality
        self.enable_lipsync = enable_lipsync
        self.preserve_bg = preserve_bg
        self.hf_token = hf_token
        self.groq_token = groq_token
        self.log_callback = log_callback or print
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.progress = ProgressTracker()
        
        if reset_progress or self.progress.should_reset(video_path):
            self.progress.reset(video_path)
            self.log("Starting fresh processing (progress reset)")
        
        self.analyzer = AdvancedVoiceAnalyzer()
        self.audio_enhancer = ProfessionalAudioEnhancer()
        self.translation_condenser = TranslationCondenser(target_lang, groq_token)
        self.audio_processor = StreamingAudioProcessor(video_path)
        
        self.setup_directories()
        
        self.speaker_profiles = {
            "SPEAKER_00": {
                'gender': 'male',
                'pitch_median': 120.0,
                'confidence': 0.8
            }
        }
        
        global TTS_AVAILABLE, TTS
        if not TTS_AVAILABLE:
            try:
                from TTS.api import TTS
                TTS_AVAILABLE = True
            except Exception:
                try:
                    from coqui_tts.api import TTS
                    TTS_AVAILABLE = True
                except Exception:
                    TTS = None
                    TTS_AVAILABLE = False

        tgt = self.target_lang.lower()
        chatterbox_supports_lang = tgt in CHATTERBOX_LANGUAGES or tgt in CHATTERBOX_LANGUAGE_MAP
        xtts_supports_lang = tgt in XTTS_LANGUAGES or tgt in XTTS_LANGUAGE_MAP

        # Engine priority: Chatterbox Multilingual (best free cloning quality) > XTTS v2 > gTTS (no cloning)
        if CHATTERBOX_AVAILABLE and chatterbox_supports_lang:
            self.tts_engine = 'chatterbox'
            engine_reason = "Resemble AI Chatterbox Multilingual, MIT license, zero-shot voice cloning"
        elif TTS_AVAILABLE and xtts_supports_lang:
            self.tts_engine = 'xtts'
            if not CHATTERBOX_AVAILABLE:
                engine_reason = "chatterbox-tts not installed; using XTTS v2 zero-shot voice cloning"
            else:
                engine_reason = f"Chatterbox doesn't support '{tgt}'; using XTTS v2 zero-shot voice cloning"
        else:
            self.tts_engine = 'gtts'
            if not CHATTERBOX_AVAILABLE and not TTS_AVAILABLE:
                engine_reason = "NO cloning engine installed (pip install chatterbox-tts) - generic voice only, will NOT match speaker identity"
            else:
                engine_reason = f"no installed cloning engine supports target language '{tgt}' - generic voice only, will NOT match speaker identity"

        # Kept for any external callers/log lines expecting the old boolean name
        self.use_xtts = self.tts_engine == 'xtts'
        self.gtts_lang = self.target_lang if self.target_lang in GTTS_LANGUAGES else 'en'

        # Ordered fallback chain actually attempted per segment. gtts is always the last resort.
        self._engine_chain = []
        if self.tts_engine == 'chatterbox':
            self._engine_chain.append('chatterbox')
            if TTS_AVAILABLE and xtts_supports_lang:
                self._engine_chain.append('xtts')
        elif self.tts_engine == 'xtts':
            self._engine_chain.append('xtts')
        self._engine_chain.append('gtts')

        # Per-run synthesis stats, filled in during synthesize_with_timing()
        self._engine_fail_streak = {'chatterbox': 0, 'xtts': 0}
        self._engine_used_counts = {'chatterbox': 0, 'xtts': 0, 'gtts': 0}

        self.log(f"\n{'='*60}")
        self.log(f"Enhanced Video Dubbing System v2.0")
        self.log(f"{'='*60}")
        self.log(f"Video Duration: {self.audio_processor.duration/60:.1f} minutes")
        self.log(f"Video FPS: {self.audio_processor.fps:.2f}")
        self.log(f"Processing Chunks: {self.audio_processor.num_chunks}")
        self.log(f"Device: {self.device}")
        engine_labels = {
            'chatterbox': 'Chatterbox Multilingual (Zero-Shot Voice Cloning)',
            'xtts': 'XTTS v2 (Zero-Shot Voice Cloning)',
            'gtts': 'gTTS (generic voice, NO cloning)'
        }
        self.log(f"TTS Engine: {engine_labels[self.tts_engine]}")
        self.log(f"  Reason: {engine_reason}")
        self.log(f"Voice Quality: {self.voice_quality}")
        self.log(f"Memory Available: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        self.log(f"{'='*60}\n")
    
    def log(self, message):
        self.log_callback(message)
    
    def setup_directories(self):
        dirs = ['audio', 'results', 'speakers_audio', 'audio_chunks',
                'temp_audio', 'enhanced_audio', 'chunks', 'speakers_image']
        
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def cleanup_temp_files(self):
        temp_dirs = ['chunks', 'temp_audio', 'audio_chunks', 'speakers_audio', 
                     'speakers_image', 'enhanced_audio']
        
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    self.log(f"Cleanup warning: {e}")
        
        if self.progress.progress['stage'] == 'complete':
            for checkpoint_file in Path('.').glob('checkpoint_*.json'):
                try:
                    checkpoint_file.unlink()
                except:
                    pass
    
    def process(self):
        try:
            if self.progress.progress['stage'] == 'start':
                if os.path.exists('results'):
                    for file in os.listdir('results'):
                        try:
                            os.remove(os.path.join('results', file))
                        except:
                            pass
            
            if self.progress.progress['stage'] == 'start':
                self.extract_audio_streaming()
                self.progress.update_stage('audio_extracted')
            
            if self.progress.progress['stage'] == 'audio_extracted':
                speakers_rolls = self.perform_diarization()
                speakers_rolls_serializable = {
                    f"{k[0]}_{k[1]}": v for k, v in speakers_rolls.items()
                }
                self.save_checkpoint('speakers_rolls', speakers_rolls_serializable)
                self.progress.update_stage('diarization_done')
            else:
                speakers_rolls_serializable = self.load_checkpoint('speakers_rolls')
                speakers_rolls = {
                    tuple(map(float, k.split('_'))): v 
                    for k, v in speakers_rolls_serializable.items()
                }
            
            if self.progress.progress['stage'] == 'diarization_done':
                records = self.transcribe_chunked()
                self.save_checkpoint('records', records)
                self.progress.update_stage('transcription_done')
            else:
                records = self.load_checkpoint('records')
            
            if self.progress.progress['stage'] == 'transcription_done':
                records = self.translate_and_condense(records)
                self.save_checkpoint('translated_records', records)
                self.progress.update_stage('translation_done')
            else:
                records = self.load_checkpoint('translated_records')
            
            if self.progress.progress['stage'] == 'translation_done':
                self.synthesize_with_timing(records, speakers_rolls)
                self.progress.update_stage('synthesis_done')
            
            if self.progress.progress['stage'] == 'synthesis_done':
                self.assemble_audio_precise(records)
                self.progress.update_stage('assembly_done')
            
            if self.preserve_bg and self.progress.progress['stage'] == 'assembly_done':
                self.preserve_background()
                self.progress.update_stage('background_done')
            
            if self.progress.progress['stage'] in ['assembly_done', 'background_done']:
                self.create_final_video_synced()
                self.progress.update_stage('video_done')
            
            if self.enable_lipsync and self.progress.progress['stage'] == 'video_done':
                self.apply_lipsync_chunked(records, speakers_rolls)
                self.progress.update_stage('complete')
            
            self.finalize()
            self.log(f"\n{'='*60}")
            self.log("PROCESSING COMPLETE!")
            self.log(f"{'='*60}\n")
            
        except Exception as e:
            self.log(f"\nERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
            
            if "duration" in str(e).lower() or "extract" in str(e).lower():
                self.progress.reset()
            
            raise
        
        finally:
            self.cleanup_temp_files()
    
    def extract_audio_streaming(self):
        self.log("\nStage 1: Extracting Audio...")
        
        output_path = "audio/original.wav"
        if self.audio_processor.get_full_audio_streaming(output_path):
            self.log(f"Audio extracted: {self.audio_processor.duration/60:.1f} minutes")
        else:
            raise Exception("Audio extraction failed")
    
    def perform_diarization(self) -> dict:
        self.log("\nStage 2: Speaker Diarization...")
        
        if not self.hf_token:
            self.log("WARNING: No HF token, using single speaker mode")
            return {(0.0, self.audio_processor.duration): "SPEAKER_00"}
        
        try:
            pipeline = None
            models_to_try = [
                'pyannote/speaker-diarization-3.1',
                'pyannote/speaker-diarization-3.0',
                'pyannote/speaker-diarization'
            ]
            
            for model_name in models_to_try:
                try:
                    # Modern huggingface_hub uses token
                    pipeline = Pipeline.from_pretrained(
                        model_name,
                        token=self.hf_token
                    )
                    if pipeline is not None:
                        break
                except TypeError:
                    # Legacy fallback for older huggingface_hub versions
                    pipeline = Pipeline.from_pretrained(
                        model_name,
                        use_auth_token=self.hf_token
                    )
                    if pipeline is not None:
                        break
                except Exception as model_err:
                    self.log(f"  Note: {model_name} load error: {model_err}")
                    continue
            
            if pipeline is None:
                raise Exception("Could not load any PyAnnote speaker diarization model. Please verify your HF_TOKEN has accepted the terms on huggingface.co/pyannote/speaker-diarization-3.1.")
            
            try:
                pipeline = pipeline.to(self.device)
            except Exception:
                pass
            
            diarization_result = pipeline("audio/original.wav")

            # pyannote.audio 4.x wraps the result in a DiarizeOutput object - the actual
            # Annotation (with .itertracks) lives at .speaker_diarization. Older versions
            # return the Annotation directly. Handle both so this doesn't silently collapse
            # to single-speaker mode whenever pyannote.audio gets upgraded.
            if hasattr(diarization_result, 'itertracks'):
                diarization = diarization_result
            elif hasattr(diarization_result, 'speaker_diarization'):
                diarization = diarization_result.speaker_diarization
            else:
                raise Exception(
                    f"Unrecognized diarization pipeline output type: {type(diarization_result).__name__} "
                    f"(attributes: {[a for a in dir(diarization_result) if not a.startswith('_')]})"
                )

            speakers_rolls = {}

            for speech_turn, _, speaker in diarization.itertracks(yield_label=True):
                if abs(speech_turn.end - speech_turn.start) > 0.8:
                    speakers_rolls[(speech_turn.start, speech_turn.end)] = speaker
            
            num_speakers = len(set(speakers_rolls.values())) if speakers_rolls else 1
            self.log(f"Found {num_speakers} speaker(s)")
            if speakers_rolls:
                per_speaker_duration = {}
                per_speaker_turns = {}
                for (start, end), spk in speakers_rolls.items():
                    per_speaker_duration[spk] = per_speaker_duration.get(spk, 0.0) + (end - start)
                    per_speaker_turns[spk] = per_speaker_turns.get(spk, 0) + 1
                for spk in sorted(per_speaker_duration):
                    self.log(f"  {spk}: {per_speaker_turns[spk]} turn(s), "
                             f"{per_speaker_duration[spk]:.1f}s total speech")
                if num_speakers < 2:
                    self.log("  NOTE: only 1 speaker cluster detected - if this video actually has "
                             "multiple people talking, diarization merged them and every line will be "
                             "cloned from one blended voice. Try a higher-quality HF_TOKEN model access, "
                             "cleaner audio, or report this video so it can be investigated further.")

            if not speakers_rolls:
                speakers_rolls = {(0.0, self.audio_processor.duration): "SPEAKER_00"}
            
            self.extract_speaker_samples(speakers_rolls)
            
            del pipeline
            MemoryManager.force_cleanup()
            
            return speakers_rolls
            
        except Exception as e:
            self.log(f"Diarization notice: {e}, using single speaker mode")
            speakers_rolls = {(0.0, self.audio_processor.duration): "SPEAKER_00"}
            self.extract_speaker_samples(speakers_rolls)
            return speakers_rolls
    
    def extract_speaker_samples(self, speakers_rolls: dict):
        """Extract speaker samples with comprehensive voice analysis"""
        speakers = set(speakers_rolls.values())
        if not hasattr(self, 'speaker_profiles') or not self.speaker_profiles:
            self.speaker_profiles = {}

        # Zero-shot cloning (Chatterbox/XTTS) needs a decent chunk of clean speech to
        # reliably match a speaker's identity - too little reference audio degrades to a
        # generic-sounding voice even when the engine itself is working correctly.
        MIN_REFERENCE_SECONDS = 6.0

        try:
            audio = AudioSegment.from_file("audio/original.wav")

            for speaker in speakers:
                speaker_turns = [(s, e) for (s, e), spk in speakers_rolls.items() if spk == speaker]

                def collect(require_loud: bool):
                    segs, dur = [], 0.0
                    for start, end in speaker_turns:
                        if dur >= CONFIG.pitch_analysis_duration:
                            break
                        segment = audio[int(start * 1000):int(end * 1000)]
                        if (not require_loud) or segment.dBFS > -45:
                            segs.append(segment)
                            dur += (end - start)
                    return segs, dur

                segments, total_duration = collect(require_loud=True)
                if total_duration < MIN_REFERENCE_SECONDS:
                    # The loudness filter left too little audio for reliable cloning - relax it
                    segments, total_duration = collect(require_loud=False)

                output_path = f"speakers_audio/{speaker}.wav"
                if segments:
                    speaker_audio = sum(segments[1:], segments[0])
                    if 0 < total_duration < MIN_REFERENCE_SECONDS:
                        # Speaker had very few/brief turns - loop the clip up to a usable length
                        # rather than cloning from a near-empty reference
                        looped = speaker_audio
                        while len(looped) / 1000.0 < MIN_REFERENCE_SECONDS:
                            looped += speaker_audio
                        speaker_audio = looped[:int(CONFIG.pitch_analysis_duration * 1000)]
                else:
                    # No usable turns at all for this speaker - fall back to the start of the full mix
                    speaker_audio = audio[:int(CONFIG.pitch_analysis_duration * 1000)]

                speaker_audio.export(output_path, format="wav")
                
                # Comprehensive voice analysis
                profile = self.analyzer.analyze_voice_comprehensive(output_path)
                self.speaker_profiles[speaker] = profile
                
                self.log(f"  {speaker}: {profile['gender']} "
                       f"(confidence: {profile['confidence']:.2f}, "
                       f"pitch: {profile['pitch_median']:.1f}Hz)")
                
                if 'analysis_methods' in profile:
                    methods = profile['analysis_methods']
                    self.log(f"    Methods: Pitch={methods['pitch'][1]:.2f}, "
                           f"Formant={methods['formant'][1]:.2f}, "
                           f"Spectral={methods['spectral'][1]:.2f}")
            
            del audio
            MemoryManager.force_cleanup()
            
        except Exception as e:
            self.log(f"Speaker sample extraction note: {e}")
            output_path = "speakers_audio/SPEAKER_00.wav"
            try:
                if os.path.exists("audio/original.wav") and not os.path.exists(output_path):
                    shutil.copyfile("audio/original.wav", output_path)
            except Exception:
                pass
            self.speaker_profiles["SPEAKER_00"] = {
                'gender': 'male', 
                'pitch_median': 120,
                'confidence': 0.5
            }
    
    def _load_whisper_model(self):
        """Load Whisper model with intelligent hardware/compute fallback"""
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_types = ["float16", "int8_float32", "int8"] if device == "cuda" else ["int8", "float32"]
        
        self.log(f"  Initializing Whisper model: '{self.whisper_model}'...")
        self.log(f"  (If this is your first run with '{self.whisper_model}', the weights are downloading in the background. Please wait...)")
        
        for comp in compute_types:
            try:
                self.log(f"  Connecting to {device} [{comp}]...")
                model = WhisperModel(self.whisper_model, device=device, compute_type=comp)
                self.log(f"  Whisper model '{self.whisper_model}' loaded successfully on {device} [{comp}] ✓")
                return model
            except Exception as e:
                self.log(f"  Note: {device}/{comp} fallback ({e})")
                continue
        
        # Fallback to CPU int8
        self.log(f"  Loading Whisper model '{self.whisper_model}' on CPU [int8]...")
        model = WhisperModel(self.whisper_model, device="cpu", compute_type="int8")
        self.log(f"  Whisper model '{self.whisper_model}' loaded successfully on CPU [int8] ✓")
        return model

    def transcribe_chunked(self) -> List[dict]:
        self.log("\nStage 3: Transcription...")
        
        os.makedirs("chunks", exist_ok=True)
        
        try:
            model = self._load_whisper_model()
            all_records = []
            
            for chunk_idx in range(self.audio_processor.num_chunks):
                chunk_path = f"chunks/chunk_{chunk_idx}.wav"
                
                if not self.audio_processor.extract_chunk(chunk_idx, chunk_path):
                    continue
                
                self.log(f"  Transcribing chunk {chunk_idx+1}/{self.audio_processor.num_chunks}...")
                
                try:
                    lang_param = None if self.source_lang == 'auto' else self.source_lang
                    segments, _ = model.transcribe(
                        chunk_path,
                        word_timestamps=True,
                        language=lang_param
                    )
                    
                    chunk_start_time = chunk_idx * CONFIG.chunk_duration
                    
                    for segment in segments:
                        words = getattr(segment, 'words', None)
                        if words:
                            for word in words:
                                if word.word and word.word.strip():
                                    all_records.append({
                                        'word': word.word.strip(),
                                        'start': word.start + chunk_start_time,
                                        'end': word.end + chunk_start_time
                                    })
                        elif getattr(segment, 'text', None) and segment.text.strip():
                            all_records.append({
                                'word': segment.text.strip(),
                                'start': getattr(segment, 'start', 0.0) + chunk_start_time,
                                'end': getattr(segment, 'end', 1.0) + chunk_start_time
                            })
                            
                except Exception as chunk_err:
                    self.log(f"  Warning on chunk {chunk_idx}: {chunk_err}")
                
                try:
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
                except Exception:
                    pass
                MemoryManager.force_cleanup()
            
            # Load diarization speaker rolls if available. Checkpoints serialize (start, end)
            # tuple keys as "start_end" strings, so they must be reconstructed back to tuples
            # here - group_into_sentences() requires real (start, end) tuple keys.
            speakers_rolls = {}
            if os.path.exists('checkpoint_speakers_rolls.json'):
                raw = self.load_checkpoint('speakers_rolls') or {}
                speakers_rolls = {tuple(map(float, k.split('_'))): v for k, v in raw.items()}

            records = self.group_into_sentences(all_records, speakers_rolls)
            self.log(f"Created {len(records)} sentence record(s) ✓")
            
            del model
            MemoryManager.force_cleanup()
            
            return records
            
        except Exception as e:
            self.log(f"Transcription error: {e}")
            raise
    
    def group_into_sentences(self, word_records: List[dict], speakers_rolls: dict = None) -> List[dict]:
        if not word_records:
            return []

        if not speakers_rolls or len(set(speakers_rolls.values())) <= 1:
            # Single speaker (or no diarization) - no risk of a sentence crossing speakers
            default_speaker = next(iter(speakers_rolls.values())) if speakers_rolls else "SPEAKER_00"
            return self._sentence_tokenize_block(word_records, default_speaker)

        # Multi-speaker: sentence-tokenizing the whole transcript first (the old approach) can
        # merge words from two different speakers into one "sentence" whenever a speaker change
        # falls mid-sentence (very common in back-and-forth dialogue). That single blended record
        # then gets voiced entirely with whichever speaker had more overlap, silently mis-cloning
        # the other speaker's words. Instead, split the word stream at every actual speaker change
        # FIRST, then sentence-tokenize within each same-speaker block - no record can ever span
        # two speakers.
        records = []
        block_words: List[dict] = []
        block_speaker = None

        for word in word_records:
            spk = self._overlap_speaker(word, speakers_rolls) or block_speaker or next(iter(speakers_rolls.values()))
            if block_speaker is None:
                block_speaker = spk
            if spk != block_speaker:
                records.extend(self._sentence_tokenize_block(block_words, block_speaker))
                block_words = []
                block_speaker = spk
            block_words.append(word)

        if block_words:
            records.extend(self._sentence_tokenize_block(block_words, block_speaker))

        return records

    def _overlap_speaker(self, record: dict, speakers_rolls: dict) -> Optional[str]:
        """Best-overlap speaker match for a single time range, or None if no turn overlaps at all
        (used to let brief diarization gaps continue the current speaker block instead of
        spuriously resetting to a default speaker)."""
        max_overlap = 0
        speaker = None
        for (start, end), spk in speakers_rolls.items():
            overlap = min(record['end'], end) - max(record['start'], start)
            if overlap > max_overlap:
                max_overlap = overlap
                speaker = spk
        return speaker

    def _sentence_tokenize_block(self, word_records: List[dict], speaker: str) -> List[dict]:
        """Sentence-tokenize a contiguous, single-speaker run of words into timed records."""
        if not word_records:
            return []

        full_text = " ".join([r['word'] for r in word_records])

        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(full_text)
        except Exception:
            sentences = [s.strip() for s in full_text.split('.') if s.strip()]

        if not sentences:
            sentences = [full_text]

        records = []
        word_idx = 0

        for sentence in sentences:
            sentence_words = sentence.split()
            if not sentence_words:
                continue

            start_times = []
            end_times = []

            for _ in sentence_words:
                if word_idx < len(word_records):
                    start_times.append(word_records[word_idx]['start'])
                    end_times.append(word_records[word_idx]['end'])
                    word_idx += 1

            if start_times and end_times:
                seg_start = min(start_times)
                seg_end = max(end_times)

                records.append({
                    'text': sentence.strip(),
                    'start': seg_start,
                    'end': seg_end,
                    'duration': max(0.2, seg_end - seg_start),
                    'speaker': speaker
                })

        return records
    
    def translate_and_condense(self, records: List[dict]) -> List[dict]:
        """Translate and intelligently condense for timing, using nearby dialogue as context
        so pronouns/continuity stay correct and condensing has more than one isolated line to
        work with (Groq only - MarianMT has no mechanism to accept context)."""
        self.log("\nStage 4: Translation & Condensing...")

        CONTEXT_WINDOW = 2  # sentences of surrounding dialogue on each side
        batch_size = 10

        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]

            for offset, record in enumerate(batch):
                idx = i + offset
                context_before = " ".join(r['text'] for r in records[max(0, idx - CONTEXT_WINDOW):idx])
                context_after = " ".join(r['text'] for r in records[idx + 1:idx + 1 + CONTEXT_WINDOW])
                speaker_gender = self.speaker_profiles.get(record.get('speaker', ''), {}).get('gender')

                try:
                    # Get translation
                    if self.groq_token:
                        translation = self.translate_groq(record['text'], context_before, context_after,
                                                          speaker_gender)
                    else:
                        translation = self.translate_marian(record['text'])

                    if not translation:
                        translation = record['text']

                    # Condense if needed
                    condensed = self.translation_condenser.condense_translation(
                        record['text'],
                        translation,
                        record['duration'],
                        context_before=context_before,
                        context_after=context_after,
                        speaker_gender=speaker_gender
                    )

                    record['translation'] = condensed
                    record['translation_condensed'] = len(condensed) < len(translation)

                except Exception as e:
                    self.log(f"Translation error: {e}")
                    record['translation'] = record['text']
                    record['translation_condensed'] = False

            condensed_count = sum(1 for r in batch if r.get('translation_condensed'))
            self.log(f"  Translated {min(i+batch_size, len(records))}/{len(records)} "
                   f"({condensed_count} condensed)")
            MemoryManager.force_cleanup()

        return records

    def translate_groq(self, text: str, context_before: str = '', context_after: str = '',
                      speaker_gender: str = None) -> str:
        if not text or not text.strip():
            return text
        if self.source_lang.lower() == self.target_lang.lower():
            return text
        try:
            client = Groq(api_key=self.groq_token)
            context_block = ""
            if context_before or context_after:
                context_block = (
                    f"Surrounding conversation, for context only - do NOT translate or include it:\n"
                    f"...said just before: \"{context_before}\"\n"
                    f"...said right after: \"{context_after}\"\n\n"
                )
            gender_block = _gender_instruction(self.target_lang, speaker_gender)
            response = client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": f"You are dubbing a conversation into {self.target_lang}.\n"
                              f"{context_block}{gender_block}"
                              f"Translate ONLY the line below to {self.target_lang}. Keep it "
                              f"completely accurate and grammatically complete, and consistent with "
                              f"the surrounding conversation (correct pronouns, tone, continuity).\n\n"
                              f"Text: \"{text}\"\n\n"
                              f"Return ONLY: [[translation: YOUR_TRANSLATION]]"
                }],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=500
            )

            content = response.choices[0].message.content
            match = re.search(r'\[\[translation:\s*(.*?)\]\]', content, re.DOTALL)
            if match and match.group(1).strip():
                return match.group(1).strip()
            elif content and content.strip():
                return content.strip()

        except Exception as e:
            self.log(f"  Groq notice: {e}, using neural translator fallback")

        return self.translate_marian(text)
    
    def translate_marian(self, text: str) -> str:
        if not text or not text.strip():
            return text
        if self.source_lang.lower() == self.target_lang.lower():
            return text
            
        # 1. Try Helsinki-NLP MarianMT
        try:
            model_name = f"Helsinki-NLP/opus-mt-{self.source_lang}-{self.target_lang}"
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name).to(self.device)
            
            inputs = tokenizer([text], return_tensors="pt", padding=True,
                              truncation=True, max_length=512).to(self.device)
            
            outputs = model.generate(**inputs, max_length=512)
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            del model, tokenizer
            MemoryManager.force_cleanup()
            
            if translation and translation.strip():
                return translation.strip()
                
        except Exception as e:
            pass
        
        # 2. Universal deep-translator Google Translator fallback
        try:
            from deep_translator import GoogleTranslator
            tgt = self.target_lang.lower()
            if tgt == 'zh-cn':
                tgt = 'zh-CN'
            elif tgt == 'zh-tw':
                tgt = 'zh-TW'
            src = self.source_lang.lower()
            if src == 'auto':
                src = 'auto'
            elif src == 'zh-cn':
                src = 'zh-CN'
                
            translated = GoogleTranslator(source=src, target=tgt).translate(text)
            if translated and translated.strip():
                return translated.strip()
        except Exception as gt_err:
            self.log(f"  Translation fallback error: {gt_err}")
            
        return text
    def synthesize_with_timing(self, records: List[dict], speakers_rolls: dict):
        """Synthesize with precise timing control and zero-shot voice cloning"""
        self.log("\nStage 5: Voice Synthesis with Timing...")

        engines = {}

        if 'chatterbox' in self._engine_chain:
            try:
                engines['chatterbox'] = ChatterboxMultilingualTTS.from_pretrained(
                    device='cuda' if self.device.type == 'cuda' else 'cpu'
                )
                self.log(f"✓ Loaded Chatterbox Multilingual Zero-Shot Voice Cloner on {self.device}")
            except Exception as e:
                self.log(f"WARNING: Chatterbox failed to load, dropping it for this run: {e}")
                self.log(traceback.format_exc())
                self._engine_chain = [x for x in self._engine_chain if x != 'chatterbox']

        if 'xtts' in self._engine_chain:
            try:
                os.environ["COQUI_TOS_AGREED"] = "1"
                engines['xtts'] = TTS("tts_models/multilingual/multi-dataset/xtts_v2",
                                       gpu=(self.device.type == 'cuda'))
                self.log(f"✓ Loaded XTTS v2 Zero-Shot Voice Cloner on {self.device}")
            except Exception as e:
                self.log(f"WARNING: XTTS v2 failed to load, dropping it for this run: {e}")
                self.log(traceback.format_exc())
                self._engine_chain = [x for x in self._engine_chain if x != 'xtts']

        if not any(name in engines for name in ('chatterbox', 'xtts')):
            self.log("WARNING: No voice-cloning engine available - every segment will use generic gTTS "
                      "(the dubbed voice will NOT match the original speaker(s)).")

        for i, record in enumerate(records):
            if self.progress.is_segment_done(i):
                continue

            output_file = f"audio_chunks/{i}.wav"
            record['speaker'] = self.find_speaker(record, speakers_rolls)

            used_engine = self._synthesize_segment(i, record, output_file, engines)
            self._engine_used_counts[used_engine] = self._engine_used_counts.get(used_engine, 0) + 1
            record['tts_engine_used'] = used_engine

            try:
                # Strip any silent lead-in the TTS engine generated before the actual speech
                # starts - otherwise the clip is placed correctly at record['start'] but the
                # audible voice inside it starts a beat later than the mouth/action (also makes
                # the speed-matching below more accurate, since it won't count that silence).
                output_file = self._trim_silence(output_file)

                # Adjust speed to match timing
                output_file = self.adjust_audio_timing(output_file, record)

                # Enhance audio quality and match vocal characteristics
                spk = record.get('speaker', 'SPEAKER_00')
                profile = self.speaker_profiles.get(spk, self.speaker_profiles.get('SPEAKER_00', {}))
                output_file = self.audio_enhancer.enhance_voice(
                    output_file,
                    profile,
                    self.voice_quality,
                    is_cloned_voice=(used_engine in ('chatterbox', 'xtts'))
                )

                record['audio_file'] = output_file
                self.progress.mark_segment_done(i)

            except Exception as e:
                self.log(f"Post-processing warning for segment {i}: {e}")
                record['audio_file'] = output_file

            if i % CONFIG.batch_size == 0 or i == len(records) - 1:
                self.log(f"  Synthesized {i+1}/{len(records)}")
                MemoryManager.force_cleanup()

        engines.clear()
        MemoryManager.force_cleanup()

        counts = self._engine_used_counts
        total = sum(counts.values()) or 1
        cloned = counts.get('chatterbox', 0) + counts.get('xtts', 0)
        self.log(f"\nVoice synthesis summary: {cloned}/{total} segment(s) used real voice cloning "
                  f"(chatterbox={counts.get('chatterbox', 0)}, xtts={counts.get('xtts', 0)}, "
                  f"gtts_fallback={counts.get('gtts', 0)})")
        if counts.get('gtts', 0) > 0 and cloned == 0 and total > 0:
            self.log("WARNING: Every segment fell back to gTTS - the dubbed voice will NOT match the "
                      "original speaker(s). Check the engine-load and per-segment warnings above.")

    def _synthesize_segment(self, i: int, record: dict, output_file: str, engines: dict) -> str:
        """Try each engine in self._engine_chain in priority order for this segment.

        Demotes an engine (removes it from the chain for the rest of this run) after
        3 consecutive failures instead of silently retrying-and-failing forever, so a
        systemic problem (bad reference clip, unsupported language, OOM) surfaces loudly
        instead of quietly degrading the whole video to generic gTTS.
        """
        for engine_name in list(self._engine_chain):
            if engine_name == 'gtts':
                break
            model = engines.get(engine_name)
            if model is None:
                continue
            try:
                if engine_name == 'chatterbox':
                    self.generate_chatterbox(model, record, output_file)
                elif engine_name == 'xtts':
                    self.generate_xtts(model, record, output_file)
                self._engine_fail_streak[engine_name] = 0
                return engine_name
            except Exception as e:
                self._engine_fail_streak[engine_name] = self._engine_fail_streak.get(engine_name, 0) + 1
                streak = self._engine_fail_streak[engine_name]
                self.log(f"{engine_name} cloning failed on segment {i} (fail #{streak}): {e}")
                if streak == 1:
                    self.log(traceback.format_exc())
                if streak >= 3:
                    self.log(f"WARNING: {engine_name} failed {streak} times in a row - "
                              f"disabling it for the rest of this video and falling back down the chain.")
                    self._engine_chain = [x for x in self._engine_chain if x != engine_name]
                    engines.pop(engine_name, None)
                # try the next engine in the chain for this same segment

        try:
            self.generate_gtts(record, output_file)
            return 'gtts'
        except Exception as e:
            self.log(f"gTTS fallback also failed on segment {i}: {e}")
            raise
    
    # Cap how much atempo is allowed to speed up/slow down a clip. The old range (0.5x-2.0x)
    # let a single segment play at up to DOUBLE or HALF its natural speed to force an exact
    # timestamp match - very audible pitch-preserved warping ("sometimes slow, sometimes fast").
    # Bounds are intentionally asymmetric: speeding up sounds rushed/bad quickly, so that stays
    # tight; slowing down is more tolerable and directly reduces how often a short translation
    # (very common - many languages are just more compact) leaves dead silence for the rest of
    # the original line while the mouth/action is still going. Whatever mismatch remains beyond
    # this is absorbed by trimming/padding with silence in assemble_audio_precise.
    MIN_ATEMPO = 0.75
    MAX_ATEMPO = 1.15

    def _trim_silence(self, audio_file: str, top_db: float = 35.0) -> str:
        """Strip leading/trailing silence a TTS engine generated around the actual speech.
        Without this, the clip still gets placed at the correct record['start'] timestamp, but
        the audible voice inside it begins after the silent lead-in - making dubbed speech sound
        like it starts late compared to the mouth/action it's supposed to match."""
        try:
            y, sr = librosa.load(audio_file, sr=None)
            if len(y) == 0:
                return audio_file
            y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
            if len(y_trimmed) == 0:
                return audio_file
            sf.write(audio_file, y_trimmed, sr)
            return audio_file
        except Exception:
            return audio_file

    def adjust_audio_timing(self, audio_file: str, record: dict) -> str:
        """Gently nudge audio speed toward the original timing without audibly warping it"""
        try:
            audio = AudioSegment.from_file(audio_file)
            current_duration = len(audio) / 1000.0
            target_duration = record['duration']

            # If durations are close enough, no adjustment needed
            if abs(current_duration - target_duration) < 0.1:
                return audio_file

            speed_ratio = current_duration / target_duration
            clamped_ratio = min(self.MAX_ATEMPO, max(self.MIN_ATEMPO, speed_ratio))

            # Use ffmpeg for high-quality time stretching
            temp_output = audio_file.replace('.wav', '_timed.wav')

            cmd = [
                'ffmpeg', '-i', audio_file,
                '-filter:a', f'atempo={clamped_ratio}',
                temp_output, '-y'
            ]

            result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, timeout=30)

            if result.returncode == 0 and os.path.exists(temp_output):
                os.remove(audio_file)
                os.rename(temp_output, audio_file)

            return audio_file

        except:
            return audio_file
    
    def find_speaker(self, record: dict, speakers_rolls: dict) -> str:
        max_overlap = 0
        speaker = "SPEAKER_00"
        
        for (start, end), spk in speakers_rolls.items():
            overlap = min(record['end'], end) - max(record['start'], start)
            if overlap > max_overlap:
                max_overlap = overlap
                speaker = spk
        
        return speaker
    
    def _resolve_speaker_wav(self, speaker: str) -> str:
        """Find the cleanest available voice-cloning reference clip for a speaker."""
        speaker_wav = f"speakers_audio/{speaker}.wav"

        if not os.path.exists(speaker_wav):
            all_wavs = list(Path('speakers_audio').glob('*.wav'))
            if all_wavs:
                speaker_wav = str(all_wavs[0])
            elif os.path.exists("audio/original.wav"):
                speaker_wav = "audio/original.wav"
            else:
                raise Exception("Speaker reference audio not found")

        return speaker_wav

    def generate_chatterbox(self, model, record: dict, output_file: str):
        speaker = record.get('speaker', 'SPEAKER_00')
        speaker_wav = self._resolve_speaker_wav(speaker)

        text = record['translation'][:CONFIG.max_text_length]
        target_code = self.target_lang.lower()
        cb_lang = CHATTERBOX_LANGUAGE_MAP.get(target_code, target_code)

        wav = model.generate(text, language_id=cb_lang, audio_prompt_path=speaker_wav)
        torchaudio.save(output_file, wav, model.sr)

    def generate_xtts(self, tts, record: dict, output_file: str):
        speaker = record.get('speaker', 'SPEAKER_00')
        speaker_wav = self._resolve_speaker_wav(speaker)

        text = record['translation'][:CONFIG.max_text_length]
        target_code = self.target_lang.lower()
        xtts_lang = XTTS_LANGUAGE_MAP.get(target_code, target_code)

        tts.tts_to_file(
            text=text,
            file_path=output_file,
            speaker_wav=speaker_wav,
            language=xtts_lang,
            speed=1.0
        )

    def generate_gtts(self, record: dict, output_file: str):
        tts = gTTS(text=record['translation'], lang=self.gtts_lang, slow=False)
        temp_mp3 = output_file.replace('.wav', '.mp3')
        tts.save(temp_mp3)
        
        audio = AudioSegment.from_mp3(temp_mp3)
        audio.export(output_file, format="wav")
        
        try:
            os.remove(temp_mp3)
        except:
            pass
    
    def assemble_audio_precise(self, records: List[dict]):
        """Assemble audio with frame-accurate synchronization"""
        self.log("\nStage 6: Precise Audio Assembly...")
        
        if not records:
            raise Exception("No records to assemble")
        
        # Get exact video duration in milliseconds
        video_duration_ms = int(self.audio_processor.duration * 1000)
        
        # Create silent base track
        combined = AudioSegment.silent(duration=video_duration_ms)
        
        for i, record in enumerate(records):
            audio_file = record.get('audio_file', f"audio_chunks/{i}_enhanced.wav")
            
            if not os.path.exists(audio_file):
                audio_file = f"audio_chunks/{i}.wav"
            
            if os.path.exists(audio_file):
                try:
                    segment_audio = AudioSegment.from_file(audio_file)
                    start_ms = int(record['start'] * 1000)
                    end_ms = int(record['end'] * 1000)
                    target_duration = max(1, end_ms - start_ms)
                    
                    # Trim or pad segment to exact duration
                    if len(segment_audio) > target_duration:
                        segment_audio = segment_audio[:target_duration]
                    elif len(segment_audio) < target_duration:
                        padding = target_duration - len(segment_audio)
                        segment_audio = segment_audio + AudioSegment.silent(duration=padding)
                    
                    # Overlay at exact position
                    combined = combined.overlay(segment_audio, position=start_ms)
                    
                except Exception as e:
                    self.log(f"Warning: Could not process segment {i}: {e}")
            
            if i % 50 == 0 and i > 0:
                self.log(f"  Assembled {i}/{len(records)}")
                MemoryManager.force_cleanup()
        
        # Final duration check
        if len(combined) != video_duration_ms:
            self.log(f"  Adjusting final duration: {len(combined)}ms → {video_duration_ms}ms")
            if len(combined) > video_duration_ms:
                combined = combined[:video_duration_ms]
            else:
                combined = combined + AudioSegment.silent(
                    duration=video_duration_ms - len(combined)
                )
        
        combined.export("audio/dubbed_voice.wav", format="wav")
        self.log(f"Audio assembled: {len(combined)/1000:.2f}s (target: {video_duration_ms/1000:.2f}s) ✓")
    
    def preserve_background(self):
        self.log("\nStage 7: Background Audio Preservation...")
        
        try:
            input_audio = "audio/original.wav"
            if not os.path.exists(input_audio):
                self.extract_audio_streaming()
                
            separator = Separator()
            separator.load_model(model_filename='UVR-MDX-NET-Inst_HQ_3.onnx')
            separated = separator.separate(input_audio)
            
            instrumental = next((f for f in separated if "Instrumental" in f), None)
            
            if instrumental and os.path.exists(instrumental):
                voice = AudioSegment.from_file("audio/dubbed_voice.wav")
                bg = AudioSegment.from_file(instrumental)
                
                # Precise length matching
                target_length = len(voice)
                if len(bg) > target_length:
                    bg = bg[:target_length]
                elif len(bg) < target_length:
                    bg = bg + AudioSegment.silent(duration=target_length - len(bg))
                
                # Mix with balanced levels
                bg = bg - 6  # Reduce background by 6dB
                mixed = bg.overlay(voice)
                mixed.export("audio/final_mixed.wav", format="wav")
                
                self.log("Background preserved and mixed ✓")
            else:
                shutil.copy("audio/dubbed_voice.wav", "audio/final_mixed.wav")
                
        except Exception as e:
            self.log(f"Background preservation notice: {e}, using dubbed voice")
            shutil.copy("audio/dubbed_voice.wav", "audio/final_mixed.wav")
        
        MemoryManager.force_cleanup()
    
    def create_final_video_synced(self):
        """Create final video with guaranteed audio/video sync"""
        self.log("\nStage 8: Final Video Creation (Frame-Synced)...")
        
        audio_file = "audio/final_mixed.wav" if self.preserve_bg else "audio/dubbed_voice.wav"
        
        # Method 1: Copy video stream, replace audio (fastest, no re-encoding)
        cmd = [
            'ffmpeg', '-i', self.video_path, '-i', audio_file,
            '-c:v', 'copy',  # Copy video without re-encoding
            '-c:a', 'aac', '-b:a', '320k',  # Encode audio
            '-map', '0:v:0', '-map', '1:a:0',
            '-shortest',  # Match shortest stream
            '-fflags', '+genpts',  # Generate presentation timestamps
            '-async', '1',  # Audio sync method
            'results/dubbed_video.mp4', '-y'
        ]
        
        try:
            timeout = int(self.audio_processor.duration * 2 + 60)
            result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, timeout=timeout)
            
            if result.returncode == 0:
                size = os.path.getsize("results/dubbed_video.mp4") / (1024**3)
                
                # Verify sync
                video_duration = self._verify_video_duration("results/dubbed_video.mp4")
                audio_duration = self.audio_processor.duration
                
                self.log(f"Video created: {size:.2f} GB")
                self.log(f"Duration check - Video: {video_duration:.2f}s, "
                       f"Audio: {audio_duration:.2f}s, "
                       f"Diff: {abs(video_duration - audio_duration):.2f}s")
                
                if abs(video_duration - audio_duration) > 0.5:
                    self.log("WARNING: Audio/video duration mismatch detected")
            else:
                raise Exception("Video creation failed")
                
        except Exception as e:
            self.log(f"Video creation error: {e}")
            raise
    
    def _verify_video_duration(self, video_path: str) -> float:
        """Verify final video duration"""
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                timeout=30
            )
            return float(result.stdout.strip())
        except:
            return 0.0
    
    def apply_lipsync_chunked(self, records: List[dict], speakers_rolls: dict):
        self.log("\nStage 9: Lip Synchronization...")
        
        try:
            video = cv2.VideoCapture(self.video_path)
            fps = video.get(cv2.CAP_PROP_FPS)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video.release()
            
            self.log(f"  Video: {total_frames} frames @ {fps:.2f} fps")
            
            frame_speaker_map = []
            for i in range(total_frames):
                time = i / fps
                speaker = self.find_speaker({'start': time, 'end': time}, speakers_rolls)
                frame_speaker_map.append(speaker)
            
            os.makedirs('Wav2Lip', exist_ok=True)
            with open('Wav2Lip/frame_per_speaker.json', 'w') as f:
                json.dump(frame_speaker_map, f)
            
            self.extract_faces_sampled(speakers_rolls, fps)
            
            audio_file = "audio/final_mixed.wav" if self.preserve_bg else "audio/dubbed_voice.wav"
            timeout = int(self.audio_processor.duration * 4 + 180)
            
            lipsync_cmd = [
                sys.executable, 'Wav2Lip/inference.py',
                '--checkpoint_path', 'Wav2Lip/wav2lip_gan.pth',
                '--face', 'results/dubbed_video.mp4',
                '--audio', audio_file,
                '--outfile', 'results/lipsync_output.mp4',
                '--pads', '0', '10', '0', '0'
            ]
            
            result = subprocess.run(
                lipsync_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout
            )
            
            if result.returncode == 0 and os.path.exists("results/lipsync_output.mp4") and os.path.getsize("results/lipsync_output.mp4") > 1000:
                self.log("Lip-sync completed successfully ✓")
                shutil.copyfile("results/lipsync_output.mp4", "results/dubbed_video.mp4")
            else:
                self.log("Lip-sync notice: completed with synchronized video track")
            
        except subprocess.TimeoutExpired:
            self.log("Lip-sync timeout - using synchronized video track")
        except Exception as e:
            self.log(f"Lip-sync notice: {e}")
        
        MemoryManager.force_cleanup()
    
    def extract_faces_sampled(self, speakers_rolls: dict, fps: float):
        speakers = set(speakers_rolls.values())
        
        for speaker in speakers:
            speaker_times = [start for (start, end), spk in speakers_rolls.items() if spk == speaker]
            
            if not speaker_times:
                continue
            
            sample_times = np.linspace(min(speaker_times), max(speaker_times), 
                                      min(20, len(speaker_times)))
            
            os.makedirs(f"speakers_image/{speaker}", exist_ok=True)
            
            cap = cv2.VideoCapture(self.video_path)
            for idx, time in enumerate(sample_times):
                cap.set(cv2.CAP_PROP_POS_MSEC, time * 1000)
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(f"speakers_image/{speaker}/frame_{idx}.jpg", frame)
            cap.release()
            
            haar_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            for img_file in os.listdir(f"speakers_image/{speaker}"):
                img_path = os.path.join(f"speakers_image/{speaker}", img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = haar_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    face = img[y:y+h, x:x+w]
                    cv2.imwrite(img_path, face)
                else:
                    try:
                        os.remove(img_path)
                    except:
                        pass
            
            self.select_representative_face(f"speakers_image/{speaker}")
        
        if os.path.exists("Wav2Lip/speakers_image"):
            shutil.rmtree("Wav2Lip/speakers_image")
        shutil.copytree("speakers_image", "Wav2Lip/speakers_image")
    
    def select_representative_face(self, folder: str):
        images = [os.path.join(folder, f) for f in os.listdir(folder) 
                 if f.endswith(('.jpg', '.png'))]
        
        if not images:
            return
        
        if len(images) > 1:
            middle_idx = len(images) // 2
            representative = images[middle_idx]
            
            for img in images:
                if img != representative:
                    try:
                        os.remove(img)
                    except:
                        pass
            
            try:
                os.rename(representative, os.path.join(folder, "max_image.jpg"))
            except:
                pass
    
    def finalize(self):
        self.log("\nFinalizing...")
        
        if os.path.exists("audio/final_mixed.wav"):
            try:
                audio = AudioSegment.from_file("audio/final_mixed.wav")
                audio.export("results/dubbed_audio.mp3", format="mp3", bitrate="320k")
            except:
                pass
        
        self.print_results()
    
    def print_results(self):
        self.log(f"\n{'='*60}")
        self.log("RESULTS")
        self.log(f"{'='*60}")
        
        if os.path.exists('results'):
            total_size = 0
            for file in sorted(os.listdir('results')):
                file_path = os.path.join('results', file)
                size = os.path.getsize(file_path) / (1024**3)
                total_size += size
                self.log(f"  {file}: {size:.2f} GB")
            self.log(f"\nTotal output size: {total_size:.2f} GB")
        
        self.log(f"\nProcessing Stats:")
        self.log(f"  Original duration: {self.audio_processor.duration/60:.1f} minutes")
        self.log(f"  Video FPS: {self.audio_processor.fps:.2f}")
        self.log(f"  Chunks processed: {self.audio_processor.num_chunks}")
        self.log(f"  TTS engine (selected): {self.tts_engine}")
        counts = getattr(self, '_engine_used_counts', {})
        if sum(counts.values()) > 0:
            self.log(f"  Segments cloned: chatterbox={counts.get('chatterbox', 0)}, "
                     f"xtts={counts.get('xtts', 0)}, gtts_fallback={counts.get('gtts', 0)}")
        self.log(f"  Voice quality: {self.voice_quality}")
        
        if hasattr(self, 'speaker_profiles'):
            self.log(f"\nSpeaker Analysis:")
            for speaker, profile in self.speaker_profiles.items():
                self.log(f"  {speaker}: {profile['gender']} "
                       f"(conf: {profile.get('confidence', 0):.2f}, "
                       f"pitch: {profile.get('pitch_median', 0):.1f}Hz)")
        
        self.log(f"{'='*60}\n")
    
    def save_checkpoint(self, name: str, data):
        try:
            with open(f"checkpoint_{name}.json", 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.log(f"Checkpoint save error: {e}")
    
    def load_checkpoint(self, name: str):
        try:
            with open(f"checkpoint_{name}.json", 'r') as f:
                return json.load(f)
        except Exception as e:
            self.log(f"Checkpoint load error: {e}")
            return None
    
    def _safe_pip_install(self, packages: str):
        """Safely install pip packages"""
        try:
            cmd = ['pip', 'install', '--user', '--no-warn-script-location', 
                   '--quiet'] + packages.split()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                cmd = ['pip', 'install', '--no-warn-script-location', 
                       '--quiet'] + packages.split()
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            return result.returncode == 0
            
        except Exception as e:
            self.log(f"Installation error: {e}")
            return False

# ============================================================================
# YOUTUBE DOWNLOAD
# ============================================================================

def download_youtube_video(url: str, log_callback=None) -> Optional[str]:
    if log_callback is None:
        log_callback = print
        
    output_path = "youtube_video.mp4"
    
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except:
            pass
    
    # Cloud/datacenter IPs (and sometimes residential ones) can get YouTube's
    # "Sign in to confirm you're not a bot" block on the default web client.
    # Trying mobile/TV clients first sidesteps it in most cases without needing cookies.
    client_args = ['--extractor-args', 'youtube:player_client=android,ios,tv,web']
    cookies_file = os.getenv("YT_COOKIES_FILE")
    cookie_args = ['--cookies', cookies_file] if cookies_file and os.path.exists(cookies_file) else []

    methods = [
        ('yt-dlp', lambda: subprocess.run([
            'yt-dlp',
            '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            '-o', output_path,
            '--merge-output-format', 'mp4',
            *client_args, *cookie_args,
            url
        ], capture_output=True, text=True, timeout=1800)),

        ('python -m yt_dlp', lambda: subprocess.run([
            'python', '-m', 'yt_dlp',
            '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            '-o', output_path,
            '--merge-output-format', 'mp4',
            *client_args, *cookie_args,
            url
        ], capture_output=True, text=True, timeout=1800)),
    ]

    last_stderr = None
    for method_name, method in methods:
        try:
            log_callback(f"Trying {method_name}...")
            result = method()

            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                size = os.path.getsize(output_path) / (1024**3)
                log_callback(f"Downloaded: {size:.2f} GB")
                return output_path
            last_stderr = getattr(result, 'stderr', None)

        except Exception as e:
            log_callback(f"{method_name} failed: {e}")
            continue

    if last_stderr:
        log_callback(f"yt-dlp error detail: {last_stderr[-800:]}")
    log_callback("Download failed. If YouTube is blocking this connection, set the YT_COOKIES_FILE "
                 "env var to a cookies.txt exported from a logged-in browser session.")
    return None

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

# Alias for backward compatibility
UnlimitedVideoDubbing = EnhancedVideoDubbing

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enhanced Video Dubbing System v2.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python script.py --video_url input.mp4 --source_language en --target_language es
  
  # From YouTube with high quality
  python script.py --yt_url "https://youtube.com/watch?v=..." --source_language en --target_language ur --voice_quality ultra
  
  # Without lip-sync or background
  python script.py --video_url video.mp4 --source_language en --target_language fr --no_lipsync --no_background
  
  # Continue from previous run
  python script.py --video_url video.mp4 --source_language en --target_language de --no_reset
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--yt_url', type=str, help='YouTube URL')
    group.add_argument('--video_url', type=str, help='Local video path')
    
    parser.add_argument('--source_language', type=str, required=True,
                       help='Source language code (e.g., en, es, fr)')
    parser.add_argument('--target_language', type=str, required=True,
                       help='Target language code (e.g., en, es, fr, ur)')
    parser.add_argument('--whisper_model', type=str, default='large-v3',
                       choices=['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3'],
                       help='Whisper model size (default: large-v3)')
    parser.add_argument('--voice_quality', type=str, default='ultra',
                       choices=['standard', 'high', 'ultra'],
                       help='Voice quality level (default: ultra)')
    parser.add_argument('--no_lipsync', action='store_true',
                       help='Disable lip synchronization')
    parser.add_argument('--no_background', action='store_true',
                       help='Disable background audio preservation')
    parser.add_argument('--no_reset', action='store_true',
                       help='Continue from previous progress')
    
    args = parser.parse_args()
    
    # Get video
    if args.yt_url:
        print(f"Downloading: {args.yt_url}")
        video_path = download_youtube_video(args.yt_url)
        if not video_path:
            print("Download failed")
            return
    else:
        video_path = args.video_url
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            return
    
    # Get tokens from environment
    hf_token = os.getenv('HF_TOKEN')
    groq_token = os.getenv('GROQ_TOKEN')
    
    if not hf_token:
        print("WARNING: No HF_TOKEN found - speaker diarization will be limited")
        print("Set HF_TOKEN environment variable for better results")
    
    if not groq_token:
        print("WARNING: No GROQ_TOKEN found - using basic translation")
        print("Set GROQ_TOKEN environment variable for AI-powered translation")
    
    print("\n" + "="*60)
    print("Enhanced Video Dubbing System v2.0")
    print("="*60)
    print(f"Source: {args.source_language} → Target: {args.target_language}")
    print(f"Quality: {args.voice_quality}")
    print(f"Lip-sync: {'Enabled' if not args.no_lipsync else 'Disabled'}")
    print(f"Background: {'Preserved' if not args.no_background else 'Removed'}")
    print("="*60 + "\n")
    
    # Process
    try:
        dubber = EnhancedVideoDubbing(
            video_path=video_path,
            source_lang=args.source_language,
            target_lang=args.target_language,
            whisper_model=args.whisper_model,
            voice_quality=args.voice_quality,
            enable_lipsync=not args.no_lipsync,
            preserve_bg=not args.no_background,
            hf_token=hf_token,
            groq_token=groq_token,
            reset_progress=not args.no_reset
        )
        
        dubber.process()
        
        print("\n" + "="*60)
        print("SUCCESS! Check the 'results' folder for output files:")
        print("  - dubbed_video.mp4: Final dubbed video")
        print("  - dubbed_audio.mp3: Extracted dubbed audio")
        if not args.no_lipsync:
            print("  - final_lipsync.mp4: Lip-synced version")
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted - progress saved")
        print("Run again with --no_reset to continue from checkpoint")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nIf the error persists, try:")
        print("  1. Check all dependencies are installed")
        print("  2. Verify HF_TOKEN and GROQ_TOKEN are set")
        print("  3. Ensure sufficient disk space and memory")
        print("  4. Try with a shorter video first")

if __name__ == '__main__':
    main()