#!/usr/bin/env python3
"""
recorder_app.py – Desktop GUI for audio recording + LLM summarization.

On Linux, records mic + desktop audio via ffmpeg/PulseAudio into a single mono
MP3. On Windows, records the selected microphone input via FFmpeg DirectShow.
After recording, summarizes the audio using the existing Ollama-based
transcription + summarization pipeline.

Requirements:
  - ffmpeg          (system:  sudo apt install ffmpeg)
  - PulseAudio      (usually pre-installed on Ubuntu/Fedora desktops)
  - Python 3.10+
  - Ollama          (https://ollama.com/download)  — for summarization
  - WhisperX + pyannote.audio                      — for transcription

Environment variables (all optional):
    RECORDER_MIC       – PulseAudio source for microphone (Linux override)
    RECORDER_MONITOR   – PulseAudio source for desktop audio monitor (Linux override)
    RECORDER_OUTPUT_DIR – base directory for recorder outputs (default: ~/recordings)

Usage:
  python recorder_app.py
"""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from transcriptor_bot.ffmpeg import (
    detect_linux_pulse_devices,
    ensure_ffmpeg_on_path,
    list_pulseaudio_sources,
    list_windows_dshow_audio_devices,
)


# ── Load .env into os.environ (no python-dotenv dependency) ─────────────────

def _load_dotenv(path: Path | None = None) -> None:
    """Parse a .env file and inject variables into os.environ (if not already set)."""
    candidates = [
        path,
        Path(__file__).resolve().parent / ".env",
        Path.home() / ".env",
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            with candidate.open(encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
            break  # stop after first found file


_load_dotenv()

FFMPEG_EXECUTABLE = ensure_ffmpeg_on_path()
IS_WINDOWS = sys.platform.startswith("win")
DEFAULT_MIC_DEVICE, DEFAULT_DESKTOP_MONITOR = detect_linux_pulse_devices()


# ── Configuration ────────────────────────────────────────────────────────────

MIC_DEVICE = os.environ.get(
    "RECORDER_MIC",
    DEFAULT_MIC_DEVICE,
)

DESKTOP_MONITOR = os.environ.get(
    "RECORDER_MONITOR",
    DEFAULT_DESKTOP_MONITOR,
)

RECORDINGS_DIR = Path(
    os.environ.get("RECORDER_OUTPUT_DIR", str(Path.home() / "recordings"))
)
AUDIO_RECORDINGS_DIR = RECORDINGS_DIR / "audio"

_REPO_ROOT = Path(__file__).resolve().parent
_SUMMARIZE_SCRIPT = _REPO_ROOT / "scripts" / "summarize_transcript.py"


# ── Application states ──────────────────────────────────────────────────────

class State:
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPED = "stopped"
    SUMMARIZING = "summarizing"


# ── Color palette (GitHub-dark inspired) ─────────────────────────────────────

_C = {
    "bg":         "#0d1117",
    "surface":    "#161b22",
    "surface2":   "#21262d",
    "border":     "#30363d",
    "fg":         "#e6edf3",
    "fg_dim":     "#8b949e",
    "green":      "#238636",
    "green_hv":   "#2ea043",
    "red":        "#b91c1c",
    "red_hv":     "#ef4444",
    "blue":       "#1d4ed8",
    "blue_hv":    "#3b82f6",
    "amber":      "#92400e",
    "amber_hv":   "#d97706",
    "purple":     "#6d28d9",
    "purple_hv":  "#7c3aed",
    "timer_rec":  "#f85149",
    "timer_pau":  "#e3b341",
    "timer_idle": "#8b949e",
}

# ── Main GUI ─────────────────────────────────────────────────────────────────

class RecorderApp:
    """Tkinter-based meeting recorder with post-recording LLM summarization."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Meeting Recorder")
        self.root.geometry("860x720")
        self.root.minsize(680, 560)
        self.root.configure(bg=_C["bg"])

        # Internal state
        self._state = State.IDLE
        self._ffmpeg: subprocess.Popen | None = None
        self._current_file: Path | None = None
        self._rec_start: float = 0.0
        self._pause_total: float = 0.0
        self._pause_mark: float = 0.0
        self._timer_id: str | None = None
        self._msg_q: queue.Queue[tuple[str, str | None]] = queue.Queue()
        self._ffmpeg_stderr_buf: list[str] = []

        # Speaker settings
        self._min_speakers_var = tk.IntVar(value=2)
        self._max_speakers_var = tk.IntVar(value=3)
        self._mic_device_var = tk.StringVar(value=os.environ.get("RECORDER_MIC", ""))

        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        AUDIO_RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

        self._build_ui()
        if IS_WINDOWS:
            self._refresh_windows_audio_devices(initial=True)
        self._apply_state(State.IDLE)
        self._poll_queue()

    # ── UI helpers ───────────────────────────────────────────────────────

    def _make_btn(
        self,
        parent: tk.Widget,
        text: str,
        command,
        bg: str,
        hv: str,
        font=("sans-serif", 11, "bold"),
        pady: int = 9,
        padx: int = 6,
        **kw,
    ) -> tk.Button:
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg="#ffffff",
            activebackground=hv,
            activeforeground="#ffffff",
            disabledforeground=_C["fg_dim"],
            font=font,
            bd=0,
            relief=tk.FLAT,
            cursor="hand2",
            pady=pady,
            padx=padx,
            **kw,
        )
        return btn

    def _set_btn(
        self, btn: tk.Button, enabled: bool, bg_on: str, hv_on: str
    ) -> None:
        if enabled:
            btn.config(state=tk.NORMAL, bg=bg_on, activebackground=hv_on, cursor="hand2")
        else:
            btn.config(state=tk.DISABLED, bg=_C["surface2"], cursor="")

    # ── UI layout ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── Header bar ──────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=_C["surface"], height=62)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)

        tk.Label(
            hdr, text="🎙  Meeting Recorder",
            bg=_C["surface"], fg=_C["fg"],
            font=("sans-serif", 13, "bold"), padx=18,
        ).pack(side=tk.LEFT, fill=tk.Y)

        # status pill (right side of header)
        self._status_var = tk.StringVar(value="Ready")
        self._status_lbl = tk.Label(
            hdr, textvariable=self._status_var,
            bg=_C["surface"], fg=_C["fg_dim"],
            font=("sans-serif", 11), padx=18,
        )
        self._status_lbl.pack(side=tk.RIGHT, fill=tk.Y)

        # timer (center of header)
        self._timer_var = tk.StringVar(value="00:00:00")
        self._timer_lbl = tk.Label(
            hdr, textvariable=self._timer_var,
            bg=_C["surface"], fg=_C["timer_idle"],
            font=("monospace", 24, "bold"),
        )
        self._timer_lbl.pack(side=tk.LEFT, expand=True)

        # ── Header separator ────────────────────────────────────────────
        tk.Frame(self.root, bg=_C["border"], height=1).pack(fill=tk.X)

        # ── Body ────────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=_C["bg"])
        body.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

        # ── Recording controls ──────────────────────────────────────────
        ctrl = tk.Frame(body, bg=_C["bg"])
        ctrl.pack(fill=tk.X, pady=(0, 10))

        kw = dict(width=12)
        self._btn_start = self._make_btn(
            ctrl, "⏺  Start", self._on_start, _C["green"], _C["green_hv"], **kw,
        )
        self._btn_start.pack(side=tk.LEFT, padx=(0, 6))

        self._btn_pause = self._make_btn(
            ctrl, "⏸  Pause", self._on_pause, _C["amber"], _C["amber_hv"], **kw,
        )
        self._btn_pause.pack(side=tk.LEFT, padx=(0, 6))

        self._btn_resume = self._make_btn(
            ctrl, "▶  Resume", self._on_resume, _C["blue"], _C["blue_hv"], **kw,
        )
        self._btn_resume.pack(side=tk.LEFT, padx=(0, 6))

        self._btn_stop = self._make_btn(
            ctrl, "⏹  Stop", self._on_stop, _C["red"], _C["red_hv"], **kw,
        )
        self._btn_stop.pack(side=tk.LEFT)

        # ── Speaker settings ────────────────────────────────────────────
        spk = tk.Frame(body, bg=_C["surface"], padx=14, pady=8)
        spk.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            spk, text="Speakers", bg=_C["surface"], fg=_C["fg_dim"],
            font=("sans-serif", 9),
        ).pack(side=tk.LEFT, padx=(0, 14))

        for label, var in (("Min", self._min_speakers_var), ("Max", self._max_speakers_var)):
            tk.Label(
                spk, text=f"{label}:", bg=_C["surface"], fg=_C["fg"],
                font=("sans-serif", 10),
            ).pack(side=tk.LEFT, padx=(0, 4))
            sb = tk.Spinbox(
                spk, textvariable=var,
                from_=1, to=10, width=3,
                bg=_C["surface2"], fg=_C["fg"],
                insertbackground=_C["fg"],
                buttonbackground=_C["border"],
                disabledbackground=_C["surface2"],
                relief=tk.FLAT, bd=1,
                font=("sans-serif", 10),
                justify=tk.CENTER,
            )
            sb.pack(side=tk.LEFT, padx=(0, 14))

        if IS_WINDOWS:
            audio = tk.Frame(body, bg=_C["surface"], padx=14, pady=10)
            audio.pack(fill=tk.X, pady=(0, 10))

            tk.Label(
                audio, text="Microphone", bg=_C["surface"], fg=_C["fg_dim"],
                font=("sans-serif", 9),
            ).pack(anchor=tk.W)

            row = tk.Frame(audio, bg=_C["surface"])
            row.pack(fill=tk.X, pady=(6, 0))

            self._mic_combo = ttk.Combobox(
                row,
                textvariable=self._mic_device_var,
                state="normal",
                width=62,
            )
            self._mic_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

            self._btn_refresh_audio = self._make_btn(
                row,
                "Refresh",
                self._refresh_windows_audio_devices,
                _C["blue"],
                _C["blue_hv"],
                width=10,
                pady=7,
            )
            self._btn_refresh_audio.pack(side=tk.LEFT, padx=(8, 0))

            self._audio_hint_var = tk.StringVar(
                value="Choose the FFmpeg microphone name shown by Windows."
            )
            tk.Label(
                audio,
                textvariable=self._audio_hint_var,
                bg=_C["surface"],
                fg=_C["fg_dim"],
                font=("sans-serif", 9),
                justify=tk.LEFT,
                anchor=tk.W,
                wraplength=780,
            ).pack(fill=tk.X, pady=(6, 0))

        # ── File path ───────────────────────────────────────────────────
        self._file_var = tk.StringVar()
        tk.Label(
            body, textvariable=self._file_var,
            bg=_C["bg"], fg=_C["fg_dim"],
            font=("monospace", 9), anchor=tk.W,
            wraplength=820, justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(0, 6))

        pick = tk.Frame(body, bg=_C["bg"])
        pick.pack(fill=tk.X, pady=(0, 8))
        self._btn_pick_recording = self._make_btn(
            pick,
            "📂  Select Recording to Summarize",
            self._on_pick_recording,
            _C["blue"],
            _C["blue_hv"],
            width=28,
            font=("sans-serif", 10, "bold"),
            pady=7,
        )
        self._btn_pick_recording.pack(side=tk.LEFT)

        # ── Summarize button (shown only when stopped) ───────────────────
        self._sum_frame = tk.Frame(body, bg=_C["bg"])
        self._btn_summarize = self._make_btn(
            self._sum_frame, "📝  Summarize Recording",
            self._on_summarize, _C["purple"], _C["purple_hv"],
            width=32, font=("sans-serif", 12, "bold"), pady=11,
        )
        self._btn_summarize.pack(pady=(0, 6))

        # ── Output log separator ────────────────────────────────────────
        self._log_sep = tk.Frame(body, bg=_C["border"], height=1)
        self._log_sep.pack(fill=tk.X, pady=(4, 6))

        tk.Label(
            body, text="Log", bg=_C["bg"], fg=_C["fg_dim"],
            font=("sans-serif", 9), anchor=tk.W,
        ).pack(fill=tk.X)

        # ── Output log ──────────────────────────────────────────────────
        self._output = scrolledtext.ScrolledText(
            body,
            wrap=tk.WORD,
            font=("monospace", 10),
            height=14,
            state=tk.DISABLED,
            bg=_C["surface"],
            fg=_C["fg"],
            insertbackground=_C["fg"],
            selectbackground="#264f78",
            relief=tk.FLAT,
            bd=0,
            padx=10,
            pady=10,
        )
        self._output.pack(fill=tk.BOTH, expand=True)

    # ── State management ─────────────────────────────────────────────────

    def _apply_state(self, state: str) -> None:
        self._state = state
        idle = state == State.IDLE
        rec = state == State.RECORDING
        pau = state == State.PAUSED
        stp = state == State.STOPPED
        smz = state == State.SUMMARIZING

        self._set_btn(self._btn_start,  idle or stp,  _C["green"],  _C["green_hv"])
        self._set_btn(self._btn_pause,  rec and not IS_WINDOWS, _C["amber"],  _C["amber_hv"])
        self._set_btn(self._btn_resume, pau and not IS_WINDOWS, _C["blue"],   _C["blue_hv"])
        self._set_btn(self._btn_stop,   rec or pau,    _C["red"],    _C["red_hv"])
        pick_enabled = state in (State.IDLE, State.STOPPED)
        self._set_btn(
            self._btn_pick_recording,
            pick_enabled,
            _C["blue"],
            _C["blue_hv"],
        )
        if IS_WINDOWS and hasattr(self, "_btn_refresh_audio"):
            refresh_enabled = state in (State.IDLE, State.STOPPED)
            self._set_btn(
                self._btn_refresh_audio,
                refresh_enabled,
                _C["blue"],
                _C["blue_hv"],
            )
            self._mic_combo.config(state="normal" if refresh_enabled else "disabled")

        # Timer colour
        if rec:
            self._timer_lbl.config(fg=_C["timer_rec"])
        elif pau:
            self._timer_lbl.config(fg=_C["timer_pau"])
        else:
            self._timer_lbl.config(fg=_C["timer_idle"])

        if stp:
            self._sum_frame.pack(fill=tk.X, pady=(4, 4), before=self._log_sep)
            self._btn_summarize.config(
                state=tk.NORMAL,
                bg=_C["purple"], activebackground=_C["purple_hv"],
                text="📝  Summarize Recording", cursor="hand2",
            )
        elif smz:
            self._sum_frame.pack(fill=tk.X, pady=(4, 4), before=self._log_sep)
            self._btn_summarize.config(
                state=tk.DISABLED,
                bg=_C["surface2"],
                text="⏳  Summarizing…", cursor="",
            )
        else:
            self._sum_frame.pack_forget()

        labels = {
            State.IDLE: "Ready",
            State.RECORDING: "🔴  Recording…",
            State.PAUSED: "⏸  Paused",
            State.STOPPED: "✅  Saved",
            State.SUMMARIZING: "🔄  Summarizing…",
        }
        self._status_var.set(labels.get(state, ""))

    # ── Timer ────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        if self._state != State.RECORDING:
            return
        elapsed = time.monotonic() - self._rec_start - self._pause_total
        h, rem = divmod(int(max(elapsed, 0)), 3600)
        m, s = divmod(rem, 60)
        self._timer_var.set(f"{h:02d}:{m:02d}:{s:02d}")
        self._timer_id = self.root.after(500, self._tick)

    def _stop_tick(self) -> None:
        if self._timer_id is not None:
            self.root.after_cancel(self._timer_id)
            self._timer_id = None

    # ── Recording controls ───────────────────────────────────────────────

    def _validate_linux_audio_sources(self) -> None:
        if IS_WINDOWS:
            return

        missing_variables: list[str] = []
        if not MIC_DEVICE:
            missing_variables.append("RECORDER_MIC")
        if not DESKTOP_MONITOR:
            missing_variables.append("RECORDER_MONITOR")

        available_sources = list_pulseaudio_sources()
        if missing_variables:
            lines = [
                "Linux recording requires PulseAudio sources for microphone and desktop monitor.",
                f"Missing: {', '.join(missing_variables)}.",
            ]
            if available_sources:
                lines.append("Available PulseAudio sources:")
                lines.extend(f"- {source}" for source in available_sources)
            else:
                lines.append("No PulseAudio sources were discovered. Check that PulseAudio/PipeWire is running.")
            raise ValueError("\n".join(lines))

        unavailable_sources = [
            source
            for source in (MIC_DEVICE, DESKTOP_MONITOR)
            if available_sources and source not in available_sources
        ]
        if unavailable_sources:
            lines = [
                "Configured PulseAudio sources were not found.",
                *[f"- {source}" for source in unavailable_sources],
                "Available PulseAudio sources:",
                *[f"- {source}" for source in available_sources],
            ]
            raise ValueError("\n".join(lines))

    def _refresh_windows_audio_devices(self, initial: bool = False) -> None:
        devices = list_windows_dshow_audio_devices()
        self._mic_combo["values"] = devices

        current_value = self._mic_device_var.get().strip()
        if current_value and current_value in devices:
            selected = current_value
        elif devices:
            selected = devices[0]
            if initial and MIC_DEVICE in devices:
                selected = MIC_DEVICE
        else:
            selected = current_value

        self._mic_device_var.set(selected)

        if devices:
            self._audio_hint_var.set(
                "Select the microphone used for recording. System audio capture is not auto-configured on Windows."
            )
            if not initial:
                self._append(f"🔄  Found {len(devices)} Windows audio input(s).\n")
        else:
            self._audio_hint_var.set(
                "No DirectShow audio inputs were found. Check whether the microphone is connected and enabled in Windows."
            )
            if not initial:
                self._append("⚠  No Windows audio inputs were found by FFmpeg.\n")

    def _build_recording_command(self, output_path: Path) -> list[str]:
        if IS_WINDOWS:
            mic_device = self._mic_device_var.get().strip()
            if not mic_device:
                raise ValueError("Select a microphone before starting the recording.")
            return [
                FFMPEG_EXECUTABLE,
                "-f", "dshow",
                "-i", f"audio={mic_device}",
                "-ac", "1",
                "-q:a", "2",
                str(output_path),
            ]

        self._validate_linux_audio_sources()
        return [
            FFMPEG_EXECUTABLE,
            "-f", "pulse", "-i", MIC_DEVICE,
            "-f", "pulse", "-i", DESKTOP_MONITOR,
            "-filter_complex", "amerge=inputs=2",
            "-ac", "1",
            "-q:a", "2",
            str(output_path),
        ]

    def _stop_ffmpeg_process(self, timeout: float = 10.0) -> None:
        if not self._ffmpeg or self._ffmpeg.poll() is not None:
            return

        try:
            if self._ffmpeg.stdin:
                self._ffmpeg.stdin.write("q\n")
                self._ffmpeg.stdin.flush()
            self._ffmpeg.wait(timeout=timeout)
        except (BrokenPipeError, OSError, ValueError):
            self._ffmpeg.terminate()
        except subprocess.TimeoutExpired:
            self._ffmpeg.kill()
            self._ffmpeg.wait()

    def _on_start(self) -> None:
        now = datetime.now()
        stamp = now.strftime('%Y-%m-%d_%H-%M-%S')
        name = f"recording_{stamp}.mp3"
        path = AUDIO_RECORDINGS_DIR / name

        # Guarantee no overwrite
        n = 1
        while path.exists():
            path = AUDIO_RECORDINGS_DIR / (
                f"recording_{stamp}_{n}.mp3"
            )
            n += 1
        self._current_file = path

        try:
            cmd = self._build_recording_command(self._current_file)
        except ValueError as exc:
            messagebox.showerror("Error", str(exc))
            return

        try:
            self._ffmpeg = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            messagebox.showerror(
                "Error",
                "ffmpeg not found.\nRun: pip install -r requirements.txt\nor install FFmpeg system-wide.",
            )
            return
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to launch ffmpeg:\n{exc}")
            return

        # Drain stderr in background to prevent pipe-buffer deadlock
        self._ffmpeg_stderr_buf = []
        threading.Thread(
            target=self._drain_ffmpeg_stderr, daemon=True,
        ).start()

        self._rec_start = time.monotonic()
        self._pause_total = 0.0
        self._file_var.set(f"📁  {self._current_file}")
        self._clear_output()
        self._append(f"▶  Recording started → {self._current_file.name}\n")
        if IS_WINDOWS:
            self._append(f"🎤  Microphone: {self._mic_device_var.get().strip()}\n")
        self._apply_state(State.RECORDING)
        self._tick()

        # Schedule a check to detect early ffmpeg crash
        self.root.after(1500, self._check_ffmpeg_alive)

    def _on_pick_recording(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select Recording",
            initialdir=str(AUDIO_RECORDINGS_DIR),
            filetypes=[
                ("Audio files", "*.mp3 *.wav *.m4a *.flac *.ogg"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            return

        picked = Path(selected)
        if not picked.exists() or picked.stat().st_size <= 0:
            messagebox.showerror("Invalid file", "Selected file does not exist or is empty.")
            return

        self._current_file = picked
        self._file_var.set(f"📁  {self._current_file}")
        self._append(f"📂  Selected recording for summarization → {picked.name}\n")
        self._apply_state(State.STOPPED)

    def _drain_ffmpeg_stderr(self) -> None:
        """Read ffmpeg stderr in background to avoid pipe-buffer deadlock."""
        try:
            assert self._ffmpeg is not None and self._ffmpeg.stderr is not None
            for line in self._ffmpeg.stderr:
                self._ffmpeg_stderr_buf.append(line)
        except (ValueError, OSError):
            pass

    def _check_ffmpeg_alive(self) -> None:
        """Detect early ffmpeg crash (e.g. wrong PulseAudio device name)."""
        if self._state not in (State.RECORDING, State.PAUSED):
            return
        if self._ffmpeg and self._ffmpeg.poll() is not None:
            rc = self._ffmpeg.returncode
            self._stop_tick()
            err = "".join(self._ffmpeg_stderr_buf[-30:])
            self._append(f"\n❌  ffmpeg exited unexpectedly (code {rc}).\n")
            if err.strip():
                self._append(f"\n{err}\n")
            self._ffmpeg = None
            self._current_file = None
            self._apply_state(State.IDLE)
        elif self._ffmpeg:
            self.root.after(2000, self._check_ffmpeg_alive)

    def _on_pause(self) -> None:
        if IS_WINDOWS:
            messagebox.showinfo("Pause unavailable", "Pause/resume is not supported on Windows in this recorder yet.")
            return
        if self._ffmpeg and self._ffmpeg.poll() is None:
            os.kill(self._ffmpeg.pid, signal.SIGSTOP)
        self._pause_mark = time.monotonic()
        self._stop_tick()
        self._append("⏸  Paused\n")
        self._apply_state(State.PAUSED)

    def _on_resume(self) -> None:
        if IS_WINDOWS:
            return
        if self._ffmpeg and self._ffmpeg.poll() is None:
            os.kill(self._ffmpeg.pid, signal.SIGCONT)
        self._pause_total += time.monotonic() - self._pause_mark
        self._append("▶  Resumed\n")
        self._apply_state(State.RECORDING)
        self._tick()

    def _on_stop(self) -> None:
        self._stop_tick()
        if self._ffmpeg and self._ffmpeg.poll() is None and not IS_WINDOWS and self._state == State.PAUSED:
            os.kill(self._ffmpeg.pid, signal.SIGCONT)
            time.sleep(0.05)
        self._stop_ffmpeg_process(timeout=10)
        self._ffmpeg = None

        # Verify the file was actually saved
        if (
            self._current_file
            and self._current_file.exists()
            and self._current_file.stat().st_size > 0
        ):
            size_kb = self._current_file.stat().st_size / 1024
            self._append(
                f"⏹  Stopped – saved to {self._current_file}\n"
                f"   Size: {size_kb:.1f} KB\n"
            )
            self._apply_state(State.STOPPED)
        else:
            self._append("⚠  Recording file is missing or empty.\n")
            self._apply_state(State.IDLE)

    # ── Summarization ────────────────────────────────────────────────────

    def _on_summarize(self) -> None:
        if not self._current_file or not self._current_file.exists():
            messagebox.showerror("Error", "Recording file not found.")
            return

        self._apply_state(State.SUMMARIZING)
        self._append("\n" + "─" * 60 + "\n")
        self._append("  Starting summarization pipeline…\n")
        self._append("─" * 60 + "\n\n")

        threading.Thread(
            target=self._summarize_worker, daemon=True,
        ).start()

    def _summarize_worker(self) -> None:
        """Run the summarization script in a subprocess (background thread)."""
        summary_dir = RECORDINGS_DIR / "summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_file = summary_dir / (
            f"{self._current_file.stem}.meeting_summary.txt"
        )

        cmd = [
            sys.executable,
            str(_SUMMARIZE_SCRIPT),
            "--audio", str(self._current_file),
            "--out", str(summary_file),
            "--transcripts-dir", str(RECORDINGS_DIR / "transcripts"),
            "--min-speakers", str(self._min_speakers_var.get()),
            "--max-speakers", str(self._max_speakers_var.get()),
            "--pull-timeout", "3600",
            "--run-timeout", "3600",
            "-v",
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(_REPO_ROOT),
            )

            # Stream output line-by-line
            assert proc.stdout is not None
            for line in iter(proc.stdout.readline, ""):
                self._msg_q.put(("log", line))
            proc.wait()

            if proc.returncode == 0 and summary_file.exists():
                text = summary_file.read_text(encoding="utf-8").strip()
                self._msg_q.put(("log", "\n" + "═" * 60 + "\n"))
                self._msg_q.put(("log", "  MEETING SUMMARY\n"))
                self._msg_q.put(("log", "═" * 60 + "\n\n"))
                self._msg_q.put(("log", text + "\n"))
            elif proc.returncode == 0:
                self._msg_q.put(("log", "\n✅  Transcript saved (no summary generated).\n"))
            else:
                self._msg_q.put((
                    "log",
                    f"\n❌  Summarization failed (exit code {proc.returncode}).\n",
                ))
        except Exception as exc:
            self._msg_q.put(("log", f"\n❌  Error: {exc}\n"))
        finally:
            self._msg_q.put(("done", None))

    # ── Message queue (worker thread → main loop) ────────────────────────

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, data = self._msg_q.get_nowait()
                if kind == "log" and data is not None:
                    self._append(data)
                elif kind == "done":
                    self._apply_state(State.STOPPED)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    # ── Output helpers ───────────────────────────────────────────────────

    def _append(self, text: str) -> None:
        self._output.config(state=tk.NORMAL)
        self._output.insert(tk.END, text)
        self._output.see(tk.END)
        self._output.config(state=tk.DISABLED)

    def _clear_output(self) -> None:
        self._output.config(state=tk.NORMAL)
        self._output.delete("1.0", tk.END)
        self._output.config(state=tk.DISABLED)

    # ── Cleanup on window close ──────────────────────────────────────────

    def on_closing(self) -> None:
        if self._state in (State.RECORDING, State.PAUSED):
            if not messagebox.askokcancel(
                "Quit", "Recording in progress. Stop and quit?"
            ):
                return
        if self._ffmpeg and self._ffmpeg.poll() is None:
            if self._state == State.PAUSED and not IS_WINDOWS:
                os.kill(self._ffmpeg.pid, signal.SIGCONT)
                time.sleep(0.05)
            self._stop_ffmpeg_process(timeout=5)
        self.root.destroy()


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    root = tk.Tk()
    app = RecorderApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
