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
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def run_whisperx_transcribe(
    audio_path: str,
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "auto",
    language: Optional[str] = None,
    batch_size: int = 16,
) -> dict:
    """Transcribe *audio_path* with WhisperX and return aligned word timestamps.

    Parameters
    ----------
    audio_path:
        Path to the audio file (wav/mp3/m4a/…).
    model_size:
        Whisper model size – ``"tiny"``, ``"base"``, ``"small"``, ``"medium"``,
        ``"large-v2"``, ``"large-v3"``.  Defaults to ``"base"``.
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
    try:
        import whisperx
    except ImportError as exc:
        raise ImportError(
            "whisperx is not installed.  Run:\n"
            "  pip install git+https://github.com/m-bain/whisperx.git"
        ) from exc

    audio_path = str(Path(audio_path).resolve())
    logger.info("Loading WhisperX model '%s' on %s …", model_size, device)

    model = whisperx.load_model(
        model_size,
        device,
        compute_type=compute_type,
        language=language,
    )

    logger.info("Loading audio: %s", audio_path)
    audio = whisperx.load_audio(audio_path)

    logger.info("Running transcription (batch_size=%d) …", batch_size)
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    detected_language = result.get("language", language or "en")
    logger.info("Detected language: %s", detected_language)

    # --- word-level alignment ---
    logger.info("Loading alignment model for language '%s' …", detected_language)
    align_model, align_metadata = whisperx.load_align_model(
        language_code=detected_language,
        device=device,
    )

    logger.info("Aligning word timestamps …")
    aligned = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    # free GPU memory
    del model, align_model
    gc.collect()

    return {
        "segments": aligned.get("segments", []),
        "word_segments": aligned.get("word_segments", []),
        "language": detected_language,
    }
