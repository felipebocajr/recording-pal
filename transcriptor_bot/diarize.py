"""
diarize.py – pyannote.audio speaker diarization.

Public API
----------
run_pyannote_diarization(audio_path, hf_token, min_speakers, max_speakers)
    Returns a tuple:
        diarization  – pyannote Annotation object (also usable as an iterable of turns)
        rttm_lines   – list of RTTM-formatted strings (one per speaker turn)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# RTTM field format (NIST standard):
# SPEAKER <file_id> 1 <start> <duration> <NA> <NA> <speaker_label> <NA> <NA>
_RTTM_TEMPLATE = "SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>"


def _load_audio_for_pyannote(audio_path: str) -> dict:
    """Load audio into memory for pyannote when built-in decoding is unavailable."""
    try:
        import torchaudio
    except ImportError as exc:
        raise RuntimeError(
            "pyannote could not decode audio and torchaudio is unavailable for fallback loading. "
            "Install torchaudio or fix torchcodec."
        ) from exc

    waveform, sample_rate = torchaudio.load(audio_path)
    return {"waveform": waveform, "sample_rate": sample_rate}


def _get_annotation_from_output(diarization: object) -> object:
    """Normalize pyannote output across versions.

    Recent pyannote releases may return a `DiarizeOutput` object instead of a
    bare `Annotation`. Prefer `exclusive_speaker_diarization` for downstream
    transcript alignment and fall back to `speaker_diarization`.
    """
    if hasattr(diarization, "itertracks"):
        return diarization

    exclusive = getattr(diarization, "exclusive_speaker_diarization", None)
    if exclusive is not None and hasattr(exclusive, "itertracks"):
        return exclusive

    speaker_diarization = getattr(diarization, "speaker_diarization", None)
    if speaker_diarization is not None and hasattr(speaker_diarization, "itertracks"):
        return speaker_diarization

    raise TypeError(
        "Unsupported pyannote diarization output: expected Annotation-like object "
        "with `itertracks`, `exclusive_speaker_diarization`, or `speaker_diarization`."
    )


def run_pyannote_diarization(
    audio_path: str,
    hf_token: Optional[str] = None,
    min_speakers: int = 1,
    max_speakers: int = 3,
) -> tuple:
    """Run pyannote speaker diarization on *audio_path*.

    Parameters
    ----------
    audio_path:
        Path to the audio file.
    hf_token:
        Hugging Face access token.  If *None*, the function falls back to the
        ``HF_TOKEN`` environment variable.  A clear ``RuntimeError`` is raised
        when neither is available, so users know exactly what to fix.
    min_speakers:
        Minimum number of speakers expected in the recording.
    max_speakers:
        Maximum number of speakers expected in the recording.

    Returns
    -------
    (diarization, rttm_lines)
        *diarization* is a ``pyannote.core.Annotation`` object whose turns can
        be iterated with ``diarization.itertracks(yield_label=True)``.

        *rttm_lines* is a list of RTTM-formatted strings (``str``) that can be
        written directly to a ``.rttm`` file.

    Raises
    ------
    RuntimeError
        When ``HF_TOKEN`` is missing or the pyannote model access has not been
        granted on the Hugging Face Hub.
    ImportError
        When ``pyannote.audio`` is not installed.
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise ImportError(
            "pyannote.audio is not installed.  Run:\n"
            "  pip install pyannote.audio"
        ) from exc

    # --- resolve token ---
    token = hf_token or os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError(
            "Hugging Face token not found.\n"
            "  • Set the HF_TOKEN environment variable, or\n"
            "  • Pass hf_token='hf_…' explicitly.\n"
            "You also need to accept the pyannote model licence at:\n"
            "  https://huggingface.co/pyannote/speaker-diarization-3.1"
        )

    audio_path = str(Path(audio_path).resolve())
    file_id = Path(audio_path).stem

    logger.info("Loading pyannote diarization pipeline …")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=token,
        )
    except Exception as exc:
        msg = str(exc)
        if "gated" in msg.lower() or "403" in msg or "401" in msg or "access" in msg.lower():
            raise RuntimeError(
                "pyannote model access denied.\n"
                "Please accept the model licence at:\n"
                "  https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "  https://huggingface.co/pyannote/segmentation-3.0\n"
                f"Original error: {exc}"
            ) from exc
        raise

    # move pipeline to GPU if available
    try:
        import torch

        if torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))
            logger.info("pyannote pipeline moved to CUDA.")
    except Exception:
        pass

    logger.info(
        "Running diarization (min_speakers=%d, max_speakers=%d) …",
        min_speakers,
        max_speakers,
    )
    try:
        diarization = pipeline(
            audio_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
    except NameError as exc:
        if "AudioDecoder" not in str(exc):
            raise

        logger.warning(
            "pyannote built-in audio decoding is unavailable; retrying with torchaudio fallback."
        )
        diarization = pipeline(
            _load_audio_for_pyannote(audio_path),
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

    diarization_annotation = _get_annotation_from_output(diarization)

    # --- build RTTM lines ---
    rttm_lines: list[str] = []
    for turn, _, speaker in diarization_annotation.itertracks(yield_label=True):
        line = _RTTM_TEMPLATE.format(
            file_id=file_id,
            start=turn.start,
            duration=turn.duration,
            speaker=speaker,
        )
        rttm_lines.append(line)

    logger.info("Diarization complete – %d turns found.", len(rttm_lines))

    # free GPU memory so subsequent Ollama model load can allocate VRAM
    try:
        import gc
        import torch
        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass

    return diarization_annotation, rttm_lines
