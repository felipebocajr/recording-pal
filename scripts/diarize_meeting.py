#!/usr/bin/env python3
"""
diarize_meeting.py – CLI for WhisperX + pyannote speaker-diarized transcription.

Usage
-----
    python scripts/diarize_meeting.py \\
        --audio path/to/meeting.wav \\
        --hf-token $HF_TOKEN \\
        --min-speakers 3 \\
        --max-speakers 3 \\
        --out outputs/

Outputs (written to --out directory)
-------------------------------------
    whisperx_result.json   – full WhisperX segments + word timestamps
    diarization.rttm       – RTTM-format diarization turns
    speaker_transcript.txt – human-readable speaker-labelled transcript
    speaker_transcript.json – machine-readable list of speaker turns
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe a meeting audio file with WhisperX + pyannote diarization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--audio",
        required=True,
        metavar="PATH",
        help="Path to the input audio file (wav/mp3/m4a/…).",
    )
    parser.add_argument(
        "--hf-token",
        metavar="TOKEN",
        default=None,
        help=(
            "Hugging Face access token.  Falls back to the HF_TOKEN "
            "environment variable when not provided."
        ),
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=3,
        metavar="N",
        help="Minimum number of speakers expected in the recording.",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=3,
        metavar="N",
        help="Maximum number of speakers expected in the recording.",
    )
    parser.add_argument(
        "--out",
        default="outputs/",
        metavar="DIR",
        help="Output directory for generated artefacts.",
    )
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="WhisperX model size.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run WhisperX on.",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        choices=["auto", "float16", "int8", "float32"],
        help="Quantisation type for WhisperX inference.",
    )
    parser.add_argument(
        "--language",
        default=None,
        metavar="LANG",
        help="ISO-639-1 language code (e.g. 'en').  Auto-detect when omitted.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="WhisperX batch size.  Reduce if you hit OOM.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args(argv)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stderr,
    )


def _step_transcribe(args: argparse.Namespace, audio_path: Path, out_dir: Path) -> dict:
    """Step 1: WhisperX transcription + word alignment."""
    from transcriptor_bot.transcribe import run_whisperx_transcribe

    log = logging.getLogger("diarize_meeting.transcribe")
    log.info("Step 1/3 – WhisperX transcription …")
    result = run_whisperx_transcribe(
        audio_path=str(audio_path),
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        batch_size=args.batch_size,
    )
    wx_out = out_dir / "whisperx_result.json"
    wx_out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("WhisperX result saved → %s", wx_out)
    return result


def _step_diarize(
    args: argparse.Namespace,
    audio_path: Path,
    hf_token: str,
    out_dir: Path,
) -> tuple:
    """Step 2: pyannote speaker diarization."""
    from transcriptor_bot.diarize import run_pyannote_diarization

    log = logging.getLogger("diarize_meeting.diarize")
    log.info("Step 2/3 – pyannote diarization …")
    diarization, rttm_lines = run_pyannote_diarization(
        audio_path=str(audio_path),
        hf_token=hf_token,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )
    rttm_out = out_dir / "diarization.rttm"
    rttm_out.write_text("\n".join(rttm_lines) + "\n", encoding="utf-8")
    log.info("RTTM saved → %s", rttm_out)
    return diarization, rttm_lines


def _step_assign_and_render(
    whisperx_result: dict,
    diarization: object,
    out_dir: Path,
) -> None:
    """Step 3: assign words to speakers and write transcript files."""
    from transcriptor_bot.assign import (
        assign_words_to_speakers,
        build_speaker_turns,
        render_transcript,
        turns_to_json,
    )

    log = logging.getLogger("diarize_meeting.assign")
    log.info("Step 3/3 – assigning words to speakers …")
    assigned_words = assign_words_to_speakers(
        word_segments=whisperx_result["word_segments"],
        diarization=diarization,
    )
    turns = build_speaker_turns(assigned_words)

    txt_out = out_dir / "speaker_transcript.txt"
    txt_out.write_text(render_transcript(turns), encoding="utf-8")
    log.info("Speaker transcript (txt) saved → %s", txt_out)

    json_out = out_dir / "speaker_transcript.json"
    json_out.write_text(turns_to_json(turns), encoding="utf-8")
    log.info("Speaker transcript (json) saved → %s", json_out)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.verbose)
    log = logging.getLogger("diarize_meeting")

    # --- validate inputs ---
    audio_path = Path(args.audio)
    if not audio_path.exists():
        log.error("Audio file not found: %s", audio_path)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- resolve HF token ---
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    if not hf_token:
        log.error(
            "Hugging Face token is required.\n"
            "  • Set the HF_TOKEN environment variable, or\n"
            "  • Pass --hf-token TOKEN\n"
            "  • Accept the pyannote model licence at:\n"
            "    https://huggingface.co/pyannote/speaker-diarization-3.1"
        )
        return 1

    # Allow running from the repo root without installing the package.
    _repo_root = Path(__file__).resolve().parent.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

    whisperx_result = _step_transcribe(args, audio_path, out_dir)

    try:
        diarization, _ = _step_diarize(args, audio_path, hf_token, out_dir)
    except RuntimeError as exc:
        log.error("Diarization failed:\n%s", exc)
        return 1

    _step_assign_and_render(whisperx_result, diarization, out_dir)

    log.info("Done.  All outputs in: %s", out_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
