"""
transcribe.py – WhisperX transcription + word-level alignment.

Public API
----------
run_whisperx_transcribe(audio_path, model_size, device, compute_type, language, batch_size)
    Returns a dict with keys:
        "segments"      – list of segment dicts (start, end, text, words[])
        "word_segments" – flat list of word-level dicts (word, start, end, score)
        "language"      – detected or specified language code
"""

from __future__ import annotations

import gc
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from transcriptor_bot.ffmpeg import ensure_ffmpeg_on_path

logger = logging.getLogger(__name__)


def _default_hf_cache_root() -> Path:
    if os.name == "nt":
        base_dir = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base_dir / "huggingface"

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home) / "huggingface"

    return Path.home() / ".cache" / "huggingface"


def _configure_hf_download_env() -> Path:
    """Configure Hugging Face download behavior for the current platform."""
    cache_root = _default_hf_cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)

    # On Windows without Developer Mode, symlink-heavy Xet downloads can fail
    # with WinError 1314. Disable Xet and keep cache paths explicit.
    if os.name == "nt":
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))

    whisperx_cache = cache_root / "whisperx-models"
    whisperx_cache.mkdir(parents=True, exist_ok=True)
    return whisperx_cache


def _load_audio_with_resolved_ffmpeg(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Decode audio using the resolved FFmpeg executable path."""
    ffmpeg_executable = ensure_ffmpeg_on_path()
    cmd = [
        ffmpeg_executable,
        "-nostdin",
        "-threads",
        "0",
        "-i",
        audio_path,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-",
    ]

    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to load audio: {exc.stderr.decode(errors='replace')}") from exc

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def run_whisperx_transcribe(
    audio_path: str,
    model_size: str = "medium",
    device: str = "cuda",
    compute_type: str = "float16",
    language: Optional[str] = "pt",
    batch_size: int = 16,
) -> dict:
    """Transcribe *audio_path* with WhisperX and return aligned word timestamps.

    Parameters
    ----------
    audio_path:
        Path to the audio file (wav/mp3/m4a/…).
    model_size:
        Whisper model size – ``"tiny"``, ``"base"``, ``"small"``, ``"medium"``,
        ``"large-v2"``, ``"Medium"``.  Defaults to ``"base"``.
    device:
        ``"cpu"`` or ``"cuda"``.
    compute_type:
        Quantisation type for faster-whisper backend – ``"auto"``, ``"float16"``,
        ``"int8"``, ``"float32"``.  ``"auto"`` is safe for both CPU and GPU.
    language:
        ISO-639-1 language code (e.g. ``"en"``). Pass ``None`` to auto-detect.
    batch_size:
        Number of audio chunks processed in parallel.  Reduce if you hit OOM.

    Returns
    -------
    dict
        ``{"segments": [...], "word_segments": [...], "language": "en"}``
    """
    ensure_ffmpeg_on_path()
    download_root = _configure_hf_download_env()

    try:
        import whisperx
    except ImportError as exc:
        raise ImportError(
            "whisperx is not installed.  Run:\n"
            "  pip install git+https://github.com/m-bain/whisperx.git"
        ) from exc

    audio_path = str(Path(audio_path).resolve())
    logger.info("Loading audio: %s", audio_path)
    audio = _load_audio_with_resolved_ffmpeg(audio_path)

    model_candidates = [model_size]
    if model_size != "base":
        model_candidates.append("base")

    device_candidates = [device]
    if str(device).startswith("cuda"):
        device_candidates.append("cpu")

    last_exc: Exception | None = None
    for device_idx, attempt_device in enumerate(device_candidates):
        for model_idx, candidate in enumerate(model_candidates):
            model = None
            align_model = None
            try:
                attempt_compute_type = compute_type
                if attempt_device == "cpu" and compute_type == "float16":
                    attempt_compute_type = "int8"

                logger.info(
                    "Loading WhisperX model '%s' on %s (compute_type=%s) …",
                    candidate,
                    attempt_device,
                    attempt_compute_type,
                )
                model = whisperx.load_model(
                    candidate,
                    attempt_device,
                    compute_type=attempt_compute_type,
                    language=language,
                    download_root=str(download_root),
                )

                logger.info("Running transcription (batch_size=%d) …", batch_size)
                result = model.transcribe(audio, batch_size=batch_size, language=language)
                detected_language = result.get("language", language or "en")
                logger.info("Detected language: %s", detected_language)

                # --- word-level alignment ---
                logger.info("Loading alignment model for language '%s' …", detected_language)
                align_model, align_metadata = whisperx.load_align_model(
                    language_code=detected_language,
                    device=attempt_device,
                )

                logger.info("Aligning word timestamps …")
                aligned = whisperx.align(
                    result["segments"],
                    align_model,
                    align_metadata,
                    audio,
                    attempt_device,
                    return_char_alignments=False,
                )

                return {
                    "segments": aligned.get("segments", []),
                    "word_segments": aligned.get("word_segments", []),
                    "language": detected_language,
                }
            except Exception as exc:
                last_exc = exc
                is_last_model = model_idx == len(model_candidates) - 1
                is_last_device = device_idx == len(device_candidates) - 1
                if is_last_model and is_last_device:
                    break
                if is_last_model:
                    logger.warning(
                        "WhisperX attempts on device '%s' failed. Switching to '%s'. Last error: %s",
                        attempt_device,
                        device_candidates[device_idx + 1],
                        exc,
                    )
                else:
                    logger.warning(
                        "WhisperX model '%s' on %s failed (%s). Retrying with fallback model 'base'.",
                        candidate,
                        attempt_device,
                        exc,
                    )
            finally:
                # free memory between attempts so fallback model has a fair chance
                if model is not None:
                    del model
                if align_model is not None:
                    del align_model
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except Exception:
                    pass

    if last_exc is not None:
        if os.name == "nt" and "WinError 1314" in str(last_exc):
            raise RuntimeError(
                "WhisperX download failed with Windows permission error (WinError 1314). "
                "Enable Windows Developer Mode (or run as Administrator) and retry."
            ) from last_exc
        raise RuntimeError(
            f"WhisperX transcription failed for models {model_candidates} on devices {device_candidates}. "
            f"Last error: {last_exc}"
        ) from last_exc
    raise RuntimeError("WhisperX transcription failed unexpectedly.")
