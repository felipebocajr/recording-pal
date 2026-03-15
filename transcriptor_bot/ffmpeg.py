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


def _get_default_sink_name() -> str:
    """Return the PulseAudio default sink name, or empty string on failure."""
    proc = subprocess.run(
        ["pactl", "info"],
        capture_output=True,
        check=False,
        **_text_subprocess_kwargs(),
    )
    for line in proc.stdout.splitlines():
        if line.startswith("Default Sink:"):
            return line.split(":", 1)[1].strip()
    return ""


def _score_mic_source(name: str) -> int:
    """Return a lower-is-better priority score for a PulseAudio mic source.

    Scoring rules (lower = preferred):
    - ``mono-fallback``      → 0  (most reliable for USB headsets / gaming mics)
    - ``input`` or ``analog-stereo`` input sources → 1
    - Any other non-monitor source                 → 2
    - Monitor sources (should never be mic)        → 99
    """
    n = name.lower()
    if ".monitor" in n:
        return 99
    if "mono-fallback" in n:
        return 0
    if "input" in n or ("analog" in n and "output" not in n):
        return 1
    return 2


def detect_linux_pulse_devices() -> tuple[str, str]:
    """Return best-effort ``(mic_source, monitor_source)`` defaults on Linux.

    Mic selection priority (via :func:`_score_mic_source`):
    1. ``mono-fallback`` sources (most reliable for USB / gaming headsets).
    2. Other ``input`` / ``analog`` non-monitor sources.
    3. Any remaining non-monitor source.

    Monitor selection priority:
    1. ``<default-sink>.monitor`` — the monitor of whatever the user hears.
    2. Any other source whose name ends with ``.monitor``.
    3. Empty string (no monitor found).
    """
    sources = list_pulseaudio_sources()
    non_monitor = [s for s in sources if ".monitor" not in s]
    mic_source = min(non_monitor, key=_score_mic_source) if non_monitor else ""

    default_sink = _get_default_sink_name()
    monitor_source = ""
    if default_sink:
        preferred = f"{default_sink}.monitor"
        if preferred in sources:
            monitor_source = preferred
    if not monitor_source:
        monitor_source = next((s for s in sources if s.endswith(".monitor")), "")

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