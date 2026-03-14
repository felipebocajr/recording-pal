# Meeting Recorder & Summarizer

A Linux desktop app for **recording meetings, speaker-diarized transcription, and LLM summarization** — all from a single GUI window.

Core pipeline:
[ffmpeg + PulseAudio](https://ffmpeg.org/) (audio capture) →
[WhisperX](https://github.com/m-bain/whisperx) (ASR + word alignment) →
[pyannote.audio](https://github.com/pyannote/pyannote-audio) (speaker diarization) →
[Ollama](https://ollama.com) (local LLM summarization with `hf.co/unsloth/Qwen3.5-9B-GGUF:Q4_K_M`)

| Artifact | Description |
|---|---|
| `recording_YYYYMMDD_HHMMSS.mp3` | Mixed mic + desktop audio recording |
| `*.speaker_transcript.txt` | Human-readable, speaker-labelled transcript |
| `*.speaker_transcript.json` | Machine-readable list of speaker turns |
| `*.meeting_summary.md` | LLM-generated structured meeting summary |

---

## Prerequisites

### System packages

```bash
sudo apt install ffmpeg pulseaudio python3-tk
```

### Python environment

```bash
# Create & activate venv
python3 -m venv venv && source venv/bin/activate

# Install Python deps
pip install -r requirements.txt
pip install git+https://github.com/m-bain/whisperx.git

# For GPU (CUDA 11.8 example):
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
# For CPU only:
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Ollama (for LLM summarization)

Install from <https://ollama.com/download>, then pull the default model:

```bash
ollama pull hf.co/unsloth/Qwen3.5-9B-GGUF:Q4_K_M
```

The smaller fallback model (`qwen3.5:4b`) is auto-pulled if the primary fails, but you can pre-fetch it:

```bash
ollama pull qwen3.5:4b
```

### Hugging Face token (for pyannote diarization)

1. Create a free account at <https://huggingface.co/join>.
2. Generate an access token at <https://huggingface.co/settings/tokens> (type: **Read**).
3. Accept the licence for **both** models (one-time, free):
   - <https://huggingface.co/pyannote/speaker-diarization-3.1>
   - <https://huggingface.co/pyannote/segmentation-3.0>
4. Export the token before launching the app:

```bash
export HF_TOKEN=hf_your_token_here
```

---

## Launch

```bash
python recorder_app.py
```

---

## Audio Device Configuration

The app defaults to the following PulseAudio devices (Logitech G522 headset):

| Source | Default device name |
|---|---|
| Microphone | `alsa_input.usb-Logitech_G522_LIGHTSPEED_-_Wireless_Mode_0000000000000000-00.mono-fallback` |
| Desktop monitor | `alsa_output.usb-Logitech_G522_LIGHTSPEED_-_Wireless_Mode_0000000000000000-00.analog-stereo.monitor` |

To use different devices, set environment variables before launching:

```bash
export RECORDER_MIC="your_pulseaudio_source_name"
export RECORDER_MONITOR="your_pulseaudio_monitor_name"
export RECORDER_OUTPUT_DIR="$HOME/recordings"   # optional, default: ~/recordings
```

Find your device names with:

```bash
pactl list sources short
```

---

## Recording Controls

| Button | Available in | Action |
|---|---|---|
| **Start** | Idle / Stopped | Launches `ffmpeg`, begins recording mic + desktop mixed to mono MP3 |
| **Pause** | Recording | Sends `SIGSTOP` to freeze ffmpeg (timer stops) |
| **Resume** | Paused | Sends `SIGCONT` to continue recording |
| **Stop** | Recording / Paused | Sends `SIGINT` for graceful shutdown; saves file |
| **Summarize** | Stopped | Runs the full transcription + LLM summary pipeline on the recording |

### UI State Machine

```
Idle ──Start──▶ Recording ──Pause──▶ Paused
                    │                  │
                    │               Resume
                    │                  │
                    └──Stop──▶ Stopped ◀┘
                                  │
                               Summarize
                                  │
                                  ▼
                             Summarizing ──done──▶ Stopped
```

---

## Output

Recordings are saved to `~/recordings/` (or `RECORDER_OUTPUT_DIR`) with the format:

```
recording_YYYYMMDD_HHMMSS.mp3
```

After clicking **Summarize**, the app:
1. Transcribes the audio with WhisperX + diarizes with pyannote (requires `HF_TOKEN`).
2. Runs the Ollama LLM to generate a structured summary.
3. Streams progress in the scrollable output area.
4. Displays the final summary directly in the window.
5. Saves transcript + summary files to `~/recordings/transcripts/` and `~/recordings/summaries/`.

---

## CLI Usage

You can also run the summarization pipeline directly from the command line:

### From a transcript file

```bash
python scripts/summarize_transcript.py \
   --transcript recordings/transcripts/<your-file>.speaker_transcript.txt \
   --out recordings/summaries/meeting-summary.md
```

### From an audio file (transcribe + diarize + summarize)

```bash
python scripts/summarize_transcript.py \
   --audio recordings/recording_20260313_150000.mp3 \
   --min-speakers 1 \
   --max-speakers 3 \
   --out recordings/summaries/meeting-summary.md
```

### Custom model

```bash
python scripts/summarize_transcript.py \
   --transcript <file> \
   --model <your-ollama-model-tag> \
   --out recordings/summaries/meeting-summary.md
```

Default: `hf.co/unsloth/Qwen3.5-9B-GGUF:Q4_K_M` → fallback: `qwen3.5:4b`

---

## GPU vs CPU — Expected Runtime

| Setup | Model | ~60 min audio |
|---|---|---|
| GPU (CUDA) | `large-v2` | ~8–12 min |
| GPU (CUDA) | `base` | ~2–3 min |
| CPU only | `base` | ~20–40 min |
| CPU only | `tiny` | ~8–15 min |

> **Tip:** The diarization step (pyannote) takes a roughly fixed ~1–3 minutes
> per hour of audio regardless of CPU/GPU.

---

## Repository Structure

```
Transcriptor-bot/
├── recorder_app.py                 ← Desktop GUI (primary entry point)
├── scripts/
│   └── summarize_transcript.py     ← CLI: transcribe + summarize (also used by GUI)
├── transcriptor_bot/               ← Python package
│   ├── __init__.py
│   ├── transcribe.py   – WhisperX ASR + alignment
│   ├── diarize.py      – pyannote diarization
│   ├── assign.py       – word→speaker assignment + rendering
│   └── summarize.py    – Ollama LLM summarization
├── recordings/                     ← output (git-ignored)
│   ├── *.mp3
│   ├── transcripts/
│   └── summaries/
├── notebooks/
│   └── colab_whisperx_pyannote_diarization.ipynb  ← optional Colab notebook
├── requirements.txt
└── README.md
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `HF_TOKEN` not set / pyannote fails | `export HF_TOKEN=hf_…` before launching |
| `pyannote model access denied (403/401)` | Accept both model licences on the Hub (links above) |
| CUDA out of memory | Set `DEVICE=cpu` or use a smaller `WHISPER_MODEL` |
| `Failed to generate summary: Ollama is not installed ...` | Install Ollama: <https://ollama.com/download> |
| `Failed to generate summary: Ollama command failed ...` | Test manually: `ollama run hf.co/unsloth/Qwen3.5-9B-GGUF:Q4_K_M` |
| `ffmpeg not found` | `sudo apt install ffmpeg` |
| `_tkinter` / `python3-tk` import error | `sudo apt install python3-tk` |
| ffmpeg exits immediately after Start | Wrong PulseAudio device name — run `pactl list sources short` and set `RECORDER_MIC` / `RECORDER_MONITOR` env vars |
| WhisperX import error | `pip install git+https://github.com/m-bain/whisperx.git` |
