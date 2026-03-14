"""Helpers to locate an FFmpeg executable for local Python workflows."""

from __future__ import annotations

import locale
import os
import re
import shutil
import subprocess
from pathlib import Path


def _text_subprocess_kwargs() -> dict[str, object]:
    encoding = locale.getpreferredencoding(False) or "utf-8"
    return {
        "text": True,
        "encoding": encoding,
        "errors": "replace",
    }


def resolve_ffmpeg_executable() -> str:
    """Return an FFmpeg executable path, preferring PATH and then imageio-ffmpeg."""
    ffmpeg_on_path = shutil.which("ffmpeg")
    if ffmpeg_on_path:
        return ffmpeg_on_path

    try:
        import imageio_ffmpeg
    except ImportError:
        return "ffmpeg"

    return imageio_ffmpeg.get_ffmpeg_exe()


def ensure_ffmpeg_on_path() -> str:
    """Expose the resolved FFmpeg directory through PATH for subprocess users."""
    ffmpeg_executable = resolve_ffmpeg_executable()
    ffmpeg_dir = str(Path(ffmpeg_executable).resolve().parent)
    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    if ffmpeg_dir not in path_entries:
        os.environ["PATH"] = (
            f"{ffmpeg_dir}{os.pathsep}{current_path}" if current_path else ffmpeg_dir
        )
    return ffmpeg_executable


def list_pulseaudio_sources() -> list[str]:
    """Return PulseAudio source names available on Linux-like systems."""
    if os.name == "nt":
        return []

    proc = subprocess.run(
        ["pactl", "list", "sources", "short"],
        capture_output=True,
        check=False,
        **_text_subprocess_kwargs(),
    )
    if proc.returncode != 0:
        return []

    sources: list[str] = []
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1].strip():
            sources.append(parts[1].strip())
    return sources


def detect_linux_pulse_devices() -> tuple[str, str]:
    """Return best-effort `(mic_source, monitor_source)` defaults on Linux."""
    sources = list_pulseaudio_sources()
    mic_source = next((source for source in sources if not source.endswith(".monitor")), "")
    monitor_source = next((source for source in sources if source.endswith(".monitor")), "")
    return mic_source, monitor_source


def list_windows_dshow_audio_devices() -> list[str]:
    """Return FFmpeg DirectShow audio input device names available on Windows."""
    if os.name != "nt":
        return []

    ffmpeg_executable = ensure_ffmpeg_on_path()
    proc = subprocess.run(
        [
            ffmpeg_executable,
            "-hide_banner",
            "-list_devices",
            "true",
            "-f",
            "dshow",
            "-i",
            "dummy",
        ],
        capture_output=True,
        check=False,
        **_text_subprocess_kwargs(),
    )

    devices: list[str] = []
    seen: set[str] = set()
    pattern = re.compile(r'\[dshow @ .*?\] "([^"]+)" \(audio\)')
    output = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
    for line in output.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        device_name = match.group(1)
        if device_name not in seen:
            seen.add(device_name)
            devices.append(device_name)
    return devices