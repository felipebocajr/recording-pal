"""
transcriptor_bot – Meeting transcription & summarization package.

WhisperX (ASR) + pyannote.audio (diarization) + Ollama (LLM summary).

Typical usage (via GUI):
    python recorder_app.py

Typical usage (via CLI):
    python scripts/summarize_transcript.py --audio recording.mp3
"""

__version__ = "0.2.0"
