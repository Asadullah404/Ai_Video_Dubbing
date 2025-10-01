import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from typing import Optional
import psutil
import torch
from dotenv import load_dotenv

load_dotenv()

try:
    from video_dubbing_core import CONFIG, compute_optimal_chunk_duration
except ImportError:
    class CONFIG:
        chunk_duration = 300

# ============================================================================
# COLOR SCHEME - Modern Dark Theme
# ============================================================================

COLORS = {
    'bg_dark': '#1a1d29',           # Main background
    'bg_medium': '#252836',         # Card background
    'bg_light': '#2d3142',          # Input background
    'accent_primary': '#6c5ce7',    # Purple accent
    'accent_secondary': '#00b894',  # Green accent
    'accent_warning': '#fdcb6e',    # Yellow
    'accent_error': '#ff7675',      # Red
    'text_primary': '#ffffff',      # White text
    'text_secondary': '#a0a4b8',    # Gray text
    'border': '#3d4152',            # Border color
    'hover': '#7d70e8',             # Hover state
}

# ============================================================================
# LANGUAGE MAPPINGS
# ============================================================================

ALL_LANGUAGES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German', 
    'it': 'Italian', 'pt': 'Portuguese', 'pl': 'Polish', 'tr': 'Turkish',
    'ru': 'Russian', 'nl': 'Dutch', 'cs': 'Czech', 'ar': 'Arabic',
    'zh-cn': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean', 'hi': 'Hindi',
    'ur': 'Urdu', 'bn': 'Bengali', 'ta': 'Tamil', 'te': 'Telugu',
    'hu': 'Hungarian', 'th': 'Thai', 'vi': 'Vietnamese', 'id': 'Indonesian'
}

# ============================================================================
# ENHANCED GUI APPLICATION
# ============================================================================

class EnhancedDubbingGUI:
    """Production-level GUI with modern 16:9 design"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Video Dubbing System v4.0")
        
        # Set 16:9 aspect ratio (1280x720 for HD)
        self.root.geometry("1280x720")
        self.root.resizable(True, True)
        self.root.minsize(1280, 720)
        
        # Configure dark theme
        self.root.configure(bg=COLORS['bg_dark'])
        self.configure_styles()
        
        # Variables
        self.video_source = tk.StringVar(value="local")
        self.video_path = tk.StringVar()
        self.youtube_url = tk.StringVar()
        self.source_lang = tk.StringVar(value="en")
        self.target_lang = tk.StringVar(value="es")
        self.whisper_model = tk.StringVar(value="large-v3")
        self.voice_quality = tk.StringVar(value="ultra")
        
        # Feature toggles
        self.enable_lipsync = tk.BooleanVar(value=True)
        self.preserve_bg = tk.BooleanVar(value=True)
        self.noise_reduction = tk.BooleanVar(value=True)
        self.use_context_translation = tk.BooleanVar(value=True)
        self.auto_enhance = tk.BooleanVar(value=True)
        
        # System info
        self.ram_gb = psutil.virtual_memory().total / (1024**3)
        self.has_gpu = torch.cuda.is_available()
        self.gpu_name = torch.cuda.get_device_name(0) if self.has_gpu else "None"
        
        self.create_widgets()
        self.center_window()
    
    def configure_styles(self):
        """Configure custom ttk styles"""
        style = ttk.Style()
        
        # Configure theme colors
        style.configure('.',
            background=COLORS['bg_dark'],
            foreground=COLORS['text_primary'],
            fieldbackground=COLORS['bg_light'],
            borderwidth=0
        )
        
        # Custom frame styles
        style.configure('Card.TFrame',
            background=COLORS['bg_medium'],
            relief='flat'
        )
        
        style.configure('Dark.TFrame',
            background=COLORS['bg_dark']
        )
        
        # Label styles
        style.configure('Title.TLabel',
            background=COLORS['bg_dark'],
            foreground=COLORS['text_primary'],
            font=('Segoe UI', 24, 'bold')
        )
        
        style.configure('Subtitle.TLabel',
            background=COLORS['bg_dark'],
            foreground=COLORS['text_secondary'],
            font=('Segoe UI', 10)
        )
        
        style.configure('Header.TLabel',
            background=COLORS['bg_medium'],
            foreground=COLORS['text_primary'],
            font=('Segoe UI', 11, 'bold')
        )
        
        style.configure('Info.TLabel',
            background=COLORS['bg_medium'],
            foreground=COLORS['text_secondary'],
            font=('Segoe UI', 9)
        )
        
        style.configure('Value.TLabel',
            background=COLORS['bg_medium'],
            foreground=COLORS['accent_secondary'],
            font=('Segoe UI', 9, 'bold')
        )
        
        # Button styles - Fixed for visibility
        style.configure('Primary.TButton',
            font=('Segoe UI', 10, 'bold'),
            padding=(20, 10)
        )
        
        style.map('Primary.TButton',
            background=[('!active', COLORS['accent_primary']), ('active', COLORS['hover'])],
            foreground=[('!active', 'white'), ('active', 'white')]
        )
        
        style.configure('Secondary.TButton',
            font=('Segoe UI', 9),
            padding=(15, 8)
        )
        
        style.map('Secondary.TButton',
            background=[('!active', COLORS['bg_light']), ('active', COLORS['border'])],
            foreground=[('!active', 'white'), ('active', 'white')]
        )
        
        # Radiobutton and Checkbutton
        style.configure('Custom.TRadiobutton',
            background=COLORS['bg_medium'],
            foreground=COLORS['text_primary'],
            font=('Segoe UI', 9)
        )
        
        style.configure('Custom.TCheckbutton',
            background=COLORS['bg_medium'],
            foreground=COLORS['text_primary'],
            font=('Segoe UI', 9)
        )
        
        # Entry style
        style.configure('Custom.TEntry',
            fieldbackground=COLORS['bg_light'],
            foreground=COLORS['text_primary'],
            borderwidth=1,
            relief='solid'
        )
        
        # Combobox style - Fixed text color
        style.configure('Custom.TCombobox',
            fieldbackground=COLORS['bg_light'],
            background=COLORS['bg_light'],
            arrowcolor=COLORS['text_primary']
        )
        
        # Configure the option menu colors
        self.root.option_add('*TCombobox*Listbox.background', COLORS['bg_light'])
        self.root.option_add('*TCombobox*Listbox.foreground', COLORS['text_primary'])
        self.root.option_add('*TCombobox*Listbox.selectBackground', COLORS['accent_primary'])
        self.root.option_add('*TCombobox*Listbox.selectForeground', 'white')
        
        # Progressbar
        style.configure('Custom.Horizontal.TProgressbar',
            background=COLORS['accent_primary'],
            troughcolor=COLORS['bg_light'],
            borderwidth=0,
            thickness=8
        )
    
    def center_window(self):
        """Center window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def create_widgets(self):
        """Create all GUI widgets in 16:9 landscape layout"""
        
        # Main container with padding
        main_container = tk.Frame(self.root, bg=COLORS['bg_dark'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # ========== LEFT PANEL (60%) ==========
        left_panel = tk.Frame(main_container, bg=COLORS['bg_dark'])
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Header section
        header_frame = tk.Frame(left_panel, bg=COLORS['bg_dark'])
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = ttk.Label(
            header_frame,
            text="Video Dubbing Studio",
            style='Title.TLabel'
        )
        title_label.pack(anchor=tk.W)
        
        subtitle_label = ttk.Label(
            header_frame,
            text="Professional AI-powered video translation and dubbing",
            style='Subtitle.TLabel'
        )
        subtitle_label.pack(anchor=tk.W)
        
        # Scrollable content area
        canvas = tk.Canvas(left_panel, bg=COLORS['bg_dark'], highlightthickness=0)
        scrollbar = tk.Scrollbar(left_panel, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=COLORS['bg_dark'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Mouse wheel binding
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # === VIDEO SOURCE CARD ===
        self.create_card(scrollable_frame, "Video Source", [
            self.create_video_source_section
        ])
        
        # === LANGUAGE SETTINGS CARD ===
        self.create_card(scrollable_frame, "Language Settings", [
            self.create_language_section
        ])
        
        # === MODEL CONFIGURATION CARD ===
        self.create_card(scrollable_frame, "Model Configuration", [
            self.create_model_section
        ])
        
        # === ADVANCED FEATURES CARD ===
        self.create_card(scrollable_frame, "Advanced Features", [
            self.create_features_section
        ])
        
        # === ACTION BUTTONS ===
        action_frame = tk.Frame(scrollable_frame, bg=COLORS['bg_dark'])
        action_frame.pack(fill=tk.X, pady=(20, 0))
        
        self.start_btn = tk.Button(
            action_frame,
            text="▶ Start Processing",
            command=self.start_processing,
            bg=COLORS['accent_primary'],
            fg='white',
            activebackground=COLORS['hover'],
            activeforeground='white',
            font=('Segoe UI', 10, 'bold'),
            relief='flat',
            cursor='hand2',
            padx=20,
            pady=10,
            borderwidth=0
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Button(
            action_frame,
            text="Exit",
            command=self.root.quit,
            bg=COLORS['bg_light'],
            fg='white',
            activebackground=COLORS['border'],
            activeforeground='white',
            font=('Segoe UI', 9),
            relief='flat',
            cursor='hand2',
            padx=15,
            pady=8,
            borderwidth=0
        ).pack(side=tk.LEFT)
        
        # ========== RIGHT PANEL (40%) ==========
        right_panel = tk.Frame(main_container, bg=COLORS['bg_dark'])
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # === SYSTEM INFO CARD ===
        self.create_card(right_panel, "System Information", [
            self.create_system_info_section
        ])
        
        # === ENVIRONMENT STATUS CARD ===
        self.create_card(right_panel, "Environment Status", [
            self.create_env_status_section
        ])
        
        # === PROGRESS CARD (initially hidden) ===
        self.progress_card = self.create_card(right_panel, "Processing Status", [
            self.create_progress_section
        ])
        self.progress_card.pack_forget()  # Hide initially
    
    def create_card(self, parent, title, content_builders):
        """Create a styled card container"""
        card = tk.Frame(parent, bg=COLORS['bg_medium'], relief='flat')
        card.pack(fill=tk.X, pady=(0, 15))
        
        # Card header
        header = tk.Frame(card, bg=COLORS['bg_medium'])
        header.pack(fill=tk.X, padx=20, pady=(15, 10))
        
        ttk.Label(header, text=title, style='Header.TLabel').pack(anchor=tk.W)
        
        # Card content
        content = tk.Frame(card, bg=COLORS['bg_medium'])
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 15))
        
        for builder in content_builders:
            builder(content)
        
        return card
    
    def create_video_source_section(self, parent):
        """Create video source selection UI"""
        # Radio buttons
        radio_frame = tk.Frame(parent, bg=COLORS['bg_medium'])
        radio_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Radiobutton(
            radio_frame,
            text="📁 Local Video File",
            variable=self.video_source,
            value="local",
            style='Custom.TRadiobutton',
            command=self.toggle_source
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(
            radio_frame,
            text="🌐 YouTube URL",
            variable=self.video_source,
            value="youtube",
            style='Custom.TRadiobutton',
            command=self.toggle_source
        ).pack(anchor=tk.W, pady=2)
        
        # Local file input
        local_frame = tk.Frame(parent, bg=COLORS['bg_medium'])
        local_frame.pack(fill=tk.X, pady=(5, 5))
        
        self.path_entry = tk.Entry(
            local_frame,
            textvariable=self.video_path,
            bg=COLORS['bg_light'],
            fg=COLORS['text_primary'],
            insertbackground=COLORS['text_primary'],
            relief='flat',
            font=('Segoe UI', 9)
        )
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 10))
        
        self.browse_btn = tk.Button(
            local_frame,
            text="Browse",
            command=self.browse_video,
            bg=COLORS['bg_light'],
            fg='white',
            activebackground=COLORS['border'],
            activeforeground='white',
            font=('Segoe UI', 9),
            relief='flat',
            cursor='hand2',
            padx=15,
            pady=8,
            borderwidth=0
        )
        self.browse_btn.pack(side=tk.RIGHT)
        
        # YouTube input
        youtube_frame = tk.Frame(parent, bg=COLORS['bg_medium'])
        youtube_frame.pack(fill=tk.X)
        
        self.youtube_entry = tk.Entry(
            youtube_frame,
            textvariable=self.youtube_url,
            bg=COLORS['bg_light'],
            fg=COLORS['text_primary'],
            insertbackground=COLORS['text_primary'],
            relief='flat',
            font=('Segoe UI', 9),
            state='disabled'
        )
        self.youtube_entry.pack(fill=tk.X, ipady=8)
    
    def create_language_section(self, parent):
        """Create language selection UI"""
        # Source language
        src_frame = tk.Frame(parent, bg=COLORS['bg_medium'])
        src_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(src_frame, text="Source:", style='Info.TLabel').pack(anchor=tk.W, pady=(0, 5))
        
        lang_row = tk.Frame(src_frame, bg=COLORS['bg_medium'])
        lang_row.pack(fill=tk.X)
        
        src_combo = ttk.Combobox(
            lang_row,
            textvariable=self.source_lang,
            values=list(ALL_LANGUAGES.keys()),
            state='readonly',
            style='Custom.TCombobox',
            font=('Segoe UI', 9),
            width=12
        )
        src_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        self.src_label = ttk.Label(lang_row, text=ALL_LANGUAGES['en'], style='Value.TLabel')
        self.src_label.pack(side=tk.LEFT)
        src_combo.bind('<<ComboboxSelected>>', self.update_lang_labels)
        
        # Target language
        tgt_frame = tk.Frame(parent, bg=COLORS['bg_medium'])
        tgt_frame.pack(fill=tk.X)
        
        ttk.Label(tgt_frame, text="Target:", style='Info.TLabel').pack(anchor=tk.W, pady=(0, 5))
        
        lang_row2 = tk.Frame(tgt_frame, bg=COLORS['bg_medium'])
        lang_row2.pack(fill=tk.X)
        
        tgt_combo = ttk.Combobox(
            lang_row2,
            textvariable=self.target_lang,
            values=list(ALL_LANGUAGES.keys()),
            state='readonly',
            style='Custom.TCombobox',
            font=('Segoe UI', 9),
            width=12
        )
        tgt_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        self.tgt_label = ttk.Label(lang_row2, text=ALL_LANGUAGES['es'], style='Value.TLabel')
        self.tgt_label.pack(side=tk.LEFT)
        tgt_combo.bind('<<ComboboxSelected>>', self.update_lang_labels)
    
    def create_model_section(self, parent):
        """Create model configuration UI"""
        # Whisper model
        whisper_frame = tk.Frame(parent, bg=COLORS['bg_medium'])
        whisper_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(whisper_frame, text="Whisper Model:", style='Info.TLabel').pack(anchor=tk.W, pady=(0, 5))
        ttk.Combobox(
            whisper_frame,
            textvariable=self.whisper_model,
            values=['tiny', 'base', 'small', 'medium', 'large-v3'],
            state='readonly',
            style='Custom.TCombobox',
            font=('Segoe UI', 9),
            width=20
        ).pack(anchor=tk.W)
        
        # Voice quality
        quality_frame = tk.Frame(parent, bg=COLORS['bg_medium'])
        quality_frame.pack(fill=tk.X)
        
        ttk.Label(quality_frame, text="Voice Quality:", style='Info.TLabel').pack(anchor=tk.W, pady=(0, 5))
        ttk.Combobox(
            quality_frame,
            textvariable=self.voice_quality,
            values=['standard', 'high', 'ultra'],
            state='readonly',
            style='Custom.TCombobox',
            font=('Segoe UI', 9),
            width=20
        ).pack(anchor=tk.W)
    
    def create_features_section(self, parent):
        """Create features toggle UI"""
        features = [
            ("🎭 Lip Synchronization", self.enable_lipsync),
            ("🎵 Preserve Background Audio", self.preserve_bg),
            ("🔇 Advanced Noise Reduction", self.noise_reduction),
            ("🤖 Context-Aware Translation", self.use_context_translation),
            ("✨ Auto Voice Enhancement", self.auto_enhance)
        ]
        
        for text, var in features:
            ttk.Checkbutton(
                parent,
                text=text,
                variable=var,
                style='Custom.TCheckbutton'
            ).pack(anchor=tk.W, pady=3)
    
    def create_system_info_section(self, parent):
        """Create system information display"""
        info_items = [
            ("💾 RAM:", f"{self.ram_gb:.1f} GB"),
            ("🎮 GPU:", self.gpu_name),
            ("⚡ Chunk Size:", f"{CONFIG.chunk_duration}s ({CONFIG.chunk_duration//60} min)")
        ]
        
        for label, value in info_items:
            row = tk.Frame(parent, bg=COLORS['bg_medium'])
            row.pack(fill=tk.X, pady=3)
            
            ttk.Label(row, text=label, style='Info.TLabel').pack(side=tk.LEFT)
            ttk.Label(row, text=value, style='Value.TLabel').pack(side=tk.LEFT, padx=(10, 0))
    
    def create_env_status_section(self, parent):
        """Create environment status display"""
        hf_token = os.getenv('HF_TOKEN')
        groq_token = os.getenv('Groq_TOKEN')
        
        statuses = [
            ("HuggingFace Token:", "✓ Found" if hf_token else "✗ Missing", 
             COLORS['accent_secondary'] if hf_token else COLORS['accent_error']),
            ("Groq API Token:", "✓ Found" if groq_token else "✗ Missing",
             COLORS['accent_secondary'] if groq_token else COLORS['accent_warning'])
        ]
        
        for label, status, color in statuses:
            row = tk.Frame(parent, bg=COLORS['bg_medium'])
            row.pack(fill=tk.X, pady=3)
            
            ttk.Label(row, text=label, style='Info.TLabel').pack(side=tk.LEFT)
            
            status_label = tk.Label(
                row,
                text=status,
                bg=COLORS['bg_medium'],
                fg=color,
                font=('Segoe UI', 9, 'bold')
            )
            status_label.pack(side=tk.LEFT, padx=(10, 0))
    
    def create_progress_section(self, parent):
        """Create progress tracking UI"""
        self.progress_bar = ttk.Progressbar(
            parent,
            mode='indeterminate',
            style='Custom.Horizontal.TProgressbar'
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = tk.Label(
            parent,
            text="Ready to start...",
            bg=COLORS['bg_medium'],
            fg=COLORS['text_secondary'],
            font=('Segoe UI', 9),
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.X, pady=(0, 10))
        
        # Log text with dark theme
        self.log_text = scrolledtext.ScrolledText(
            parent,
            height=15,
            bg=COLORS['bg_dark'],
            fg=COLORS['text_primary'],
            insertbackground=COLORS['text_primary'],
            relief='flat',
            font=('Consolas', 8),
            state='disabled'
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
    
    def toggle_source(self):
        """Toggle between local file and YouTube URL"""
        if self.video_source.get() == "local":
            self.path_entry.config(state='normal')
            self.browse_btn.config(state='normal')
            self.youtube_entry.config(state='disabled')
        else:
            self.path_entry.config(state='disabled')
            self.browse_btn.config(state='disabled')
            self.youtube_entry.config(state='normal')
    
    def browse_video(self):
        """Open file browser for video selection"""
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.video_path.set(filename)
    
    def update_lang_labels(self, event=None):
        """Update language display labels"""
        self.src_label.config(text=ALL_LANGUAGES.get(self.source_lang.get(), ""))
        self.tgt_label.config(text=ALL_LANGUAGES.get(self.target_lang.get(), ""))
    
    def log(self, message):
        """Add message to log"""
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        self.root.update()
    
    def validate_inputs(self) -> Optional[dict]:
        """Validate user inputs and return config"""
        if self.video_source.get() == "local":
            video = self.video_path.get()
            if not video or not os.path.exists(video):
                messagebox.showerror("Error", "Please select a valid video file")
                return None
        else:
            video = self.youtube_url.get()
            if not video or not video.startswith(('http://', 'https://')):
                messagebox.showerror("Error", "Please enter a valid YouTube URL")
                return None
        
        if not os.getenv('HF_TOKEN'):
            if not messagebox.askyesno(
                "Warning",
                "HuggingFace token not found. Speaker diarization may fail.\n\nContinue anyway?"
            ):
                return None
        
        return {
            'video_path': video,
            'is_youtube': self.video_source.get() == "youtube",
            'source_lang': self.source_lang.get(),
            'target_lang': self.target_lang.get(),
            'whisper_model': self.whisper_model.get(),
            'voice_quality': self.voice_quality.get(),
            'enable_lipsync': self.enable_lipsync.get(),
            'preserve_bg': self.preserve_bg.get(),
            'noise_reduction': self.noise_reduction.get(),
            'use_context': self.use_context_translation.get(),
            'auto_enhance': self.auto_enhance.get()
        }
    
    def start_processing(self):
        """Start the dubbing process"""
        config = self.validate_inputs()
        if not config:
            return
        
        # Show progress card
        self.progress_card.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        self.progress_bar.start()
        self.start_btn.config(state='disabled')
        
        # Start processing thread
        thread = threading.Thread(target=self.process_video, args=(config,), daemon=True)
        thread.start()
    
    def process_video(self, config):
        """Process video in background thread"""
        try:
            self.log("="*60)
            self.log("Starting Advanced Video Dubbing System v4.0")
            self.log("="*60)
            
            if config['is_youtube']:
                self.log("\nDownloading from YouTube...")
                from video_dubbing_core import download_youtube_video
                video_path = download_youtube_video(config['video_path'], self.log)
                if not video_path:
                    self.log("ERROR: Download failed")
                    self.finish_processing(False)
                    return
            else:
                video_path = config['video_path']
            
            self.log(f"\nVideo: {video_path}")
            self.log(f"Source: {config['source_lang']} -> Target: {config['target_lang']}")
            
            from video_dubbing_core import UnlimitedVideoDubbing
            
            dubber = UnlimitedVideoDubbing(
                video_path=video_path,
                source_lang=config['source_lang'],
                target_lang=config['target_lang'],
                whisper_model=config['whisper_model'],
                voice_quality=config['voice_quality'],
                enable_lipsync=config['enable_lipsync'],
                preserve_bg=config['preserve_bg'],
                hf_token=os.getenv('HF_TOKEN'),
                groq_token=os.getenv('Groq_TOKEN') if config['use_context'] else None,
                log_callback=self.log
            )
            
            dubber.process()
            
            self.log("\n" + "="*60)
            self.log("PROCESSING COMPLETE!")
            self.log("="*60)
            
            self.finish_processing(True)
            
        except Exception as e:
            self.log(f"\nERROR: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            self.finish_processing(False)
    
    def finish_processing(self, success: bool):
        """Clean up after processing"""
        self.progress_bar.stop()
        self.start_btn.config(state='normal')
        
        if success:
            messagebox.showinfo(
                "Success",
                "Video dubbing completed successfully!\n\nCheck the 'results' folder for output files."
            )
        else:
            messagebox.showerror(
                "Error",
                "Processing failed. Check the log for details."
            )

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Launch enhanced GUI application"""
    root = tk.Tk()
    app = EnhancedDubbingGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()