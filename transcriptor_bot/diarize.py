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
            use_auth_token=token,
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
    diarization = pipeline(
        audio_path,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    # --- build RTTM lines ---
    rttm_lines: list[str] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        line = _RTTM_TEMPLATE.format(
            file_id=file_id,
            start=turn.start,
            duration=turn.duration,
            speaker=speaker,
        )
        rttm_lines.append(line)

    logger.info("Diarization complete – %d turns found.", len(rttm_lines))
    return diarization, rttm_lines
