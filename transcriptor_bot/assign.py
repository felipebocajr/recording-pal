"""
assign.py – Assign WhisperX words to diarization speakers, then render transcript.

Public API
----------
assign_words_to_speakers(word_segments, diarization)
    Returns a list of word dicts, each augmented with a ``"speaker"`` key.

build_speaker_turns(assigned_words)
    Groups consecutive same-speaker words into turn dicts.

render_transcript(turns)
    Returns a human-readable string:
        1 [SPEAKER_00 @ 00:00:01]
        Hello, this is the first speaker.

        2 [SPEAKER_01 @ 00:00:07]
        And I am the second.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core assignment logic
# ---------------------------------------------------------------------------

def assign_words_to_speakers(
    word_segments: list[dict],
    diarization: Any,
) -> list[dict]:
    """Assign each WhisperX word to the best-matching diarization speaker.

    The assignment uses **maximum overlap**: for every word whose start/end
    timestamps are known, we pick the speaker turn that overlaps most with the
    word's time span.  Words without reliable timestamps fall back to the
    speaker of the nearest turn (midpoint heuristic).

    Parameters
    ----------
    word_segments:
        Flat list of word dicts produced by WhisperX
        (``{"word": "…", "start": 0.0, "end": 0.4, "score": 0.99}``).
    diarization:
        A ``pyannote.core.Annotation`` object as returned by
        :func:`transcriptor_bot.diarize.run_pyannote_diarization`.

    Returns
    -------
    list[dict]
        Copy of *word_segments* with an added ``"speaker"`` key on each entry.
        Words for which no diarization turn could be found get
        ``"speaker": "UNKNOWN"``.
    """
    # Pre-build a list of (start, end, speaker) tuples for fast scanning.
    turns: list[tuple[float, float, str]] = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]

    assigned: list[dict] = []
    for word_info in word_segments:
        word_copy = dict(word_info)
        w_start = word_info.get("start")
        w_end = word_info.get("end")

        if w_start is None or w_end is None:
            word_copy["speaker"] = "UNKNOWN"
            assigned.append(word_copy)
            continue

        # --- max-overlap strategy ---
        best_speaker = "UNKNOWN"
        best_overlap = 0.0

        for t_start, t_end, speaker in turns:
            overlap = max(0.0, min(w_end, t_end) - max(w_start, t_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

        # --- midpoint fallback when no overlap is found ---
        if best_speaker == "UNKNOWN" and turns:
            midpoint = (w_start + w_end) / 2.0
            best_speaker = min(
                turns,
                key=lambda t: abs(((t[0] + t[1]) / 2.0) - midpoint),
            )[2]

        word_copy["speaker"] = best_speaker
        assigned.append(word_copy)

    logger.debug("Assigned %d words to speakers.", len(assigned))
    return assigned


# ---------------------------------------------------------------------------
# Group words into speaker turns
# ---------------------------------------------------------------------------

def build_speaker_turns(assigned_words: list[dict]) -> list[dict]:
    """Group consecutive same-speaker words into contiguous turn dicts.

    Parameters
    ----------
    assigned_words:
        Output of :func:`assign_words_to_speakers`.

    Returns
    -------
    list[dict]
        Each dict has keys:
        ``"speaker"``, ``"start"``, ``"end"``, ``"text"``
        (text is the words joined with spaces, leading/trailing spaces stripped).
    """
    turns: list[dict] = []
    current: dict | None = None

    for word_info in assigned_words:
        speaker = word_info.get("speaker", "UNKNOWN")
        text = word_info.get("word", "")
        w_start = word_info.get("start")
        w_end = word_info.get("end")

        if current is None or speaker != current["speaker"]:
            if current is not None:
                current["text"] = current["text"].strip()
                turns.append(current)
            current = {
                "speaker": speaker,
                "start": w_start,
                "end": w_end,
                "text": text,
            }
        else:
            current["text"] += " " + text
            if w_end is not None:
                current["end"] = w_end

    if current is not None:
        current["text"] = current["text"].strip()
        turns.append(current)

    return turns


# ---------------------------------------------------------------------------
# Render to human-readable text
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float | None) -> str:
    """Format *seconds* as ``HH:MM:SS``."""
    if seconds is None:
        return "??:??:??"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def render_transcript(turns: list[dict]) -> str:
    """Render speaker turns as a numbered, human-readable transcript.

    Output format::

        1 [SPEAKER_00 @ 00:00:01]
        Hello, this is the first speaker.

        2 [SPEAKER_01 @ 00:00:07]
        And I am the second speaker.

    Parameters
    ----------
    turns:
        Output of :func:`build_speaker_turns`.

    Returns
    -------
    str
        Plain-text transcript.
    """
    lines: list[str] = []
    for i, turn in enumerate(turns, start=1):
        ts = _fmt_time(turn.get("start"))
        lines.append(f"{i} [{turn['speaker']} @ {ts}]")
        lines.append(turn["text"])
        lines.append("")  # blank line between turns
    return "\n".join(lines).rstrip() + "\n"


def turns_to_json(turns: list[dict]) -> str:
    """Serialise *turns* to a pretty-printed JSON string."""
    return json.dumps(turns, indent=2, ensure_ascii=False)
