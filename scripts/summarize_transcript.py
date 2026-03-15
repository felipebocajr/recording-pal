#!/usr/bin/env python3
"""
summarize_transcript.py – Generate meeting summary from speaker transcript using Ollama.

Default model:
    hf.co/unsloth/Qwen3.5-9B-GGUF:Q4_K_M  (fallback: qwen3.5:4b)

Example
-------
python scripts/summarize_transcript.py \
    --transcript recordings/transcripts/meeting-123.speaker_transcript.txt \
    --out recordings/summaries/meeting-123-summary.txt
"""

from __future__ import annotations

import argparse
import os
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from repo root without installing package.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from transcriptor_bot.summarize import summarize_transcript_with_ollama


def _get_env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.environ.get(name, default or "")
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _fallback_turns_from_segments(segments: list[dict]) -> list[dict]:
    turns: list[dict] = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        turns.append(
            {
                "speaker": "UNKNOWN",
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": text,
            }
        )
    return turns


def _is_pyannote_auth_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    markers = (
        "401",
        "access denied",
        "gated repo",
        "cannot access gated repo",
        "accept the model licence",
        "accept user conditions",
        "please log in",
        "restricted",
    )
    return any(marker in msg for marker in markers)


def _transcribe_audio_to_local_transcript(
    audio_path: Path,
    transcripts_dir: Path,
    min_speakers: int,
    max_speakers: int,
) -> Path:
    from transcriptor_bot.assign import (
        assign_words_to_speakers,
        build_speaker_turns,
        render_transcript,
        turns_to_json,
    )
    from transcriptor_bot.diarize import run_pyannote_diarization
    from transcriptor_bot.transcribe import run_whisperx_transcribe

    log = logging.getLogger("summarize_transcript.transcribe")

    hf_token = _get_env("HF_TOKEN", required=False)
    model_size = _get_env("WHISPER_MODEL", "large-v3-turbo")
    default_device = "cpu" if os.name == "nt" else "cuda"
    device = _get_env("DEVICE", default_device)
    compute_type = _get_env("COMPUTE_TYPE", "float16")
    language = _get_env("WHISPER_LANGUAGE", "pt")
    batch_size = int(_get_env("BATCH_SIZE", "16"))

    log.info("Step 1/3 – WhisperX transcription from audio: %s", audio_path)
    whisperx_result = run_whisperx_transcribe(
        audio_path=str(audio_path),
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        language=language,
        batch_size=batch_size,
    )

    if not whisperx_result.get("segments"):
        raise RuntimeError("No speech was detected in the audio.")

    diarization = None
    if hf_token:
        log.info("Step 2/3 – Speaker diarization")
        try:
            diarization, _ = run_pyannote_diarization(
                audio_path=str(audio_path),
                hf_token=hf_token,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        except Exception as exc:
            if _is_pyannote_auth_error(exc):
                log.warning(
                    "pyannote diarization not authorized; continuing without speaker diarization. "
                    "Accept model licences on Hugging Face and set HF_TOKEN to enable speaker labels."
                )
            else:
                raise
    else:
        log.warning(
            "HF_TOKEN not set; skipping diarization and continuing with UNKNOWN speaker labels."
        )

    log.info("Step 3/3 – Build speaker transcript")
    word_segments = whisperx_result.get("word_segments", [])
    if word_segments and diarization is not None:
        assigned_words = assign_words_to_speakers(word_segments=word_segments, diarization=diarization)
        turns = build_speaker_turns(assigned_words)
    else:
        turns = _fallback_turns_from_segments(whisperx_result.get("segments", []))

    if not turns:
        raise RuntimeError("Failed to produce transcript text from the audio.")

    transcript_text = render_transcript(turns)
    transcript_json = turns_to_json(turns)

    transcripts_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"{audio_path.stem}-{timestamp}"

    txt_path = transcripts_dir / f"{base}.speaker_transcript.txt"
    json_path = transcripts_dir / f"{base}.speaker_transcript.json"
    txt_path.write_text(transcript_text, encoding="utf-8")
    json_path.write_text(transcript_json, encoding="utf-8")

    log.info("Transcript saved -> %s", txt_path)
    log.info("Transcript JSON saved -> %s", json_path)
    return txt_path


def _transcribe_audio_in_subprocess(
    audio_path: Path,
    transcripts_dir: Path,
    min_speakers: int,
    max_speakers: int,
    verbose: bool = False,
) -> Path:
    """Run transcription in a child process so its CUDA context is fully released
    (including the PyTorch allocator memory) before Ollama tries to load the LLM.

    The child runs this same script with ``--_transcribe-only``, prints the
    resulting transcript path prefixed by ``TRANSCRIPT_PATH:``, then exits.
    When the child exits the OS reclaims its entire CUDA context, giving Ollama
    a clean GPU.
    """
    log = logging.getLogger("summarize_transcript.transcribe_subprocess")
    cmd = [
        sys.executable,
        __file__,
        "--audio", str(audio_path),
        "--transcripts-dir", str(transcripts_dir),
        "--min-speakers", str(min_speakers),
        "--max-speakers", str(max_speakers),
        "--_transcribe-only",
    ]
    if verbose:
        cmd.append("-v")

    log.info(
        "Launching transcription subprocess to isolate CUDA context from Ollama …"
    )
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge so caller (recorder_app) sees all logs
        text=True,
    )
    assert proc.stdout is not None
    transcript_path: Path | None = None
    for line in iter(proc.stdout.readline, ""):
        if line.startswith("TRANSCRIPT_PATH:"):
            transcript_path = Path(line[len("TRANSCRIPT_PATH:"):].strip())
        else:
            # re-emit so the line appears in the parent's log stream
            sys.stderr.write(line)
            sys.stderr.flush()
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(
            f"Transcription subprocess exited with code {proc.returncode}."
        )
    if transcript_path is None:
        raise RuntimeError(
            "Transcription subprocess finished but did not report a transcript path."
        )
    log.info("Transcription subprocess finished – CUDA memory released.")
    return transcript_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a speaker transcript with Ollama local model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--transcript",
        metavar="PATH",
        help="Path to transcript file (.txt or .json).",
    )
    source_group.add_argument(
        "--audio",
        metavar="PATH",
        help="Path to input audio file. Script will transcribe first, then summarize.",
    )

    parser.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Output markdown/txt summary file path. If omitted, auto-generated.",
    )
    parser.add_argument(
        "--model",
        default="hf.co/unsloth/Qwen3.5-9B-GGUF:Q4_K_M",
        metavar="MODEL",
        help="Ollama model tag (e.g. hf.co/...:UD-Q4_K_XL).",
    )
    parser.add_argument(
        "--fallback-model",
        default="qwen3.5:4b",
        metavar="MODEL",
        help="Fallback Ollama model when the primary model fails.",
    )
    parser.add_argument(
        "--no-model-fallback",
        action="store_true",
        help="Disable model fallback and use only --model.",
    )
    parser.add_argument(
        "--no-auto-pull",
        action="store_true",
        help="Do not auto-download missing Ollama model; fail with instructions instead.",
    )
    parser.add_argument(
        "--pull-timeout",
        type=int,
        default=1800,
        metavar="SECONDS",
        help="Timeout for `ollama pull` in seconds.",
    )
    parser.add_argument(
        "--run-timeout",
        type=int,
        default=1800,
        metavar="SECONDS",
        help="Timeout for `ollama run` in seconds.",
    )
    parser.add_argument(
        "--no-cpu-fallback",
        action="store_true",
        help="Disable automatic CPU retry when GPU allocation fails.",
    )
    parser.add_argument(
        "--language",
        default="pt-BR",
        metavar="LANG",
        help="Summary language instruction (e.g. pt-BR, en).",
    )
    parser.add_argument(
        "--transcripts-dir",
        default="recordings/transcripts",
        metavar="DIR",
        help="Where to store transcript files when --audio is used.",
    )
    parser.add_argument("--min-speakers", type=int, default=2, metavar="N", help="Min speakers for diarization.")
    parser.add_argument("--max-speakers", type=int, default=3, metavar="N", help="Max speakers for diarization.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logs.")
    # Internal flag: run only the transcription step, emit TRANSCRIPT_PATH:<path>, then exit.
    # Used by _transcribe_audio_in_subprocess() to isolate CUDA context from Ollama.
    parser.add_argument("--_transcribe-only", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if verbose else logging.INFO,
        stream=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.verbose)

    # ── Internal transcribe-only mode ──────────────────────────────────────────
    # Called by _transcribe_audio_in_subprocess(); run transcription, print path,
    # exit.  The caller (parent process) reclaims the CUDA context when we exit.
    if getattr(args, "_transcribe_only", False):
        audio_path = Path(args.audio).expanduser().resolve()
        if not audio_path.exists():
            logging.getLogger("summarize_transcript").error("Audio file not found: %s", audio_path)
            return 1
        try:
            transcript_path = _transcribe_audio_to_local_transcript(
                audio_path=audio_path,
                transcripts_dir=Path(args.transcripts_dir).expanduser().resolve(),
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
            )
        except Exception as exc:
            logging.getLogger("summarize_transcript").error("Transcription failed: %s", exc)
            return 1
        # Special token the parent process scans for
        print(f"TRANSCRIPT_PATH:{transcript_path}")
        return 0

    transcript_path: Path
    if args.audio:
        audio_path = Path(args.audio).expanduser().resolve()
        if not audio_path.exists():
            logging.getLogger("summarize_transcript").error("Audio file not found: %s", audio_path)
            return 1
        try:
            transcript_path = _transcribe_audio_in_subprocess(
                audio_path=audio_path,
                transcripts_dir=Path(args.transcripts_dir).expanduser().resolve(),
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                verbose=args.verbose,
            )
        except Exception as exc:
            logging.getLogger("summarize_transcript").error("Failed during audio transcription stage: %s", exc)
            return 1
    else:
        transcript_path = Path(args.transcript).expanduser().resolve()
        if not transcript_path.exists():
            logging.getLogger("summarize_transcript").error("Transcript file not found: %s", transcript_path)
            return 1

    if args.out:
        output_path = Path(args.out).expanduser().resolve()
    else:
        summaries_dir = Path("recordings/summaries").resolve()
        summaries_dir.mkdir(parents=True, exist_ok=True)
        stem = transcript_path.stem.replace(".speaker_transcript", "")
        output_path = summaries_dir / f"{stem}.meeting_summary.txt"

    try:
        summarize_transcript_with_ollama(
            transcript_path=str(transcript_path),
            output_path=str(output_path),
            model=args.model,
            fallback_model=None if args.no_model_fallback else args.fallback_model,
            language=args.language,
            auto_pull=not args.no_auto_pull,
            pull_timeout_seconds=args.pull_timeout,
            run_timeout_seconds=args.run_timeout,
            allow_cpu_fallback=not args.no_cpu_fallback,
        )
    except Exception as exc:
        log = logging.getLogger("summarize_transcript")
        exc_str = str(exc)
        ollama_unavailable_markers = (
            "não está instalado",
            "not installed",
            "não está em execução",
            "not running",
            "ollama serve",
            "https://ollama.com/download",
        )
        if any(m in exc_str.lower() for m in ollama_unavailable_markers):
            log.warning("Ollama is unavailable – transcript saved but summary skipped. %s", exc)
            # Write a placeholder summary so the UI shows something useful.
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                "Summary unavailable - Ollama not installed\n\n"
                f"{exc}\n\n"
                "Transcript was saved successfully. To generate a summary:\n\n"
                "1. Install Ollama from https://ollama.com/download\n"
                "2. Start Ollama (`ollama serve`)\n"
                f"3. Re-run: `python scripts/summarize_transcript.py "
                f"--transcript \"{transcript_path}\"`\n",
                encoding="utf-8",
            )
            # Return 0 – transcript pipeline succeeded; summary is optional.
            return 0
        log.error("Failed to generate summary: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
