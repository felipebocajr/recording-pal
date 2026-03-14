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
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# RTTM field format (NIST standard):
# SPEAKER <file_id> 1 <start> <duration> <NA> <NA> <speaker_label> <NA> <NA>
_RTTM_TEMPLATE = "SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>"

_PYANNOTE_SAMPLE_RATE = 16_000


def _load_audio_for_pyannote(audio_path: str) -> dict:
    """Decode audio with ffmpeg subprocess and return a dict pyannote accepts.

    pyannote supports: ``{'waveform': (channels, time) float32 Tensor, 'sample_rate': int}``
    This bypasses torchaudio/torchcodec entirely on Windows where those backends
    require FFmpeg shared DLLs that are not present.
    """
    import numpy as np
    import torch

    from .ffmpeg import ensure_ffmpeg_on_path

    ffmpeg_executable = ensure_ffmpeg_on_path()
    cmd = [
        ffmpeg_executable,
        "-nostdin",
        "-threads", "0",
        "-i", audio_path,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(_PYANNOTE_SAMPLE_RATE),
        "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg failed to decode '{audio_path}': {exc.stderr.decode(errors='replace')}"
        ) from exc

    samples = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
    waveform = torch.from_numpy(samples).unsqueeze(0)  # shape: (1, time)
    return {"waveform": waveform, "sample_rate": _PYANNOTE_SAMPLE_RATE}


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
    # Always pre-load audio via ffmpeg so pyannote never needs torchaudio/torchcodec.
    # This sidesteps Windows FFmpeg DLL issues in torio/torchcodec entirely.
    try:
        audio_input = _load_audio_for_pyannote(audio_path)
        logger.debug("Audio pre-loaded via ffmpeg for pyannote (%d samples).", audio_input["waveform"].shape[-1])
    except Exception as exc:
        logger.warning("ffmpeg audio pre-load failed (%s); passing raw path to pyannote.", exc)
        audio_input = audio_path

    try:
        diarization = pipeline(
            audio_input,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
    except (NameError, RuntimeError) as exc:
        exc_str = str(exc)
        # If we passed a waveform dict and still got an error, re-raise
        if not isinstance(audio_input, str):
            raise
        # Fallback: try with pre-loaded audio when file-path approach fails
        backend_keywords = ("AudioDecoder", "backend", "torchcodec", "torchaudio", "ffmpeg")
        if not any(kw.lower() in exc_str.lower() for kw in backend_keywords):
            raise
        logger.warning(
            "pyannote audio decoding failed with file path ('%s'); retrying with pre-loaded waveform.",
            exc,
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
