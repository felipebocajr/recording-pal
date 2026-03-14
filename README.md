# Meeting Recorder & Summarizer

A desktop app for **recording meetings, speaker-diarized transcription, and LLM summarization** — all from a single GUI window.

Platform status:
- Linux: mic + desktop audio mixed recording (PulseAudio).
- Windows: microphone recording supported from the GUI device picker (desktop/system audio requires extra setup such as Stereo Mix or virtual cable).

Core pipeline:
[ffmpeg + PulseAudio](https://ffmpeg.org/) (audio capture) →
[WhisperX](https://github.com/m-bain/whisperx) (ASR + word alignment) →
[pyannote.audio](https://github.com/pyannote/pyannote-audio) (speaker diarization) →
[Ollama](https://ollama.com) (local LLM summarization with `hf.co/unsloth/Qwen3.5-9B-GGUF:Q4_K_M`)

| Artifact | Description |
|---|---|
| `recording_YYYY-MM-DD_HH-MM-SS.mp3` | Recorded audio file |
| `*.speaker_transcript.txt` | Human-readable, speaker-labelled transcript |
| `*.speaker_transcript.json` | Machine-readable list of speaker turns |
| `*.meeting_summary.txt` | LLM-generated structured meeting summary |

---

## Prerequisites

### Linux system packages

```bash
sudo apt install ffmpeg pulseaudio python3-tk
```

### Python environment (Linux/macOS)

```bash
# Create & activate venv
python3 -m venv venv
source venv/bin/activate

# Install Python deps
pip install -r requirements.txt

# Optional GPU (CUDA 11.8 example)
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Python environment (Windows PowerShell)

```powershell
# Create venv
py -m venv venv

# Activate (current shell only)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1

# If policy is blocked, use:
# .\venv\Scripts\activate.bat

# Install Python deps
python -m pip install -r requirements.txt
```

### Windows notes

- Recording in the app does not require system ffmpeg if `imageio-ffmpeg` is installed from `requirements.txt`.
- For best compatibility with Hugging Face model cache downloads, run a normal local profile path (default `C:\Users\<you>`) and keep enough free disk space.

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
4. Set the token before launching the app:

```bash
export HF_TOKEN=hf_your_token_here
```

```powershell
$env:HF_TOKEN="hf_your_token_here"
```

---

## Launch

```bash
python recorder_app.py
```

```powershell
python recorder_app.py
```

---

## Audio Device Configuration

### Linux (PulseAudio)

The app attempts to auto-detect one microphone source and one desktop monitor
source from `pactl list sources short`.

If auto-detection picks the wrong sources or cannot find them, set environment
variables before launching:

```bash
export RECORDER_MIC="your_pulseaudio_source_name"
export RECORDER_MONITOR="your_pulseaudio_monitor_name"
export RECORDER_OUTPUT_DIR="$HOME/recordings"   # optional, default: ~/recordings
```

Find your device names with:

```bash
pactl list sources short
```

The microphone source usually does not end with `.monitor`, while desktop audio
capture usually does.

### Windows (DirectShow)

- Use the in-app **Microphone** dropdown and click **Refresh** to list available devices.
- The selected device is used when you press **Start**.
- `RECORDER_MIC` can still be set manually, but UI selection is recommended on Windows.

---

## Recording Controls

| Button | Available in | Action |
|---|---|---|
| **Start** | Idle / Stopped | Launches `ffmpeg`, begins recording mic + desktop mixed to mono MP3 |
| **Pause** | Recording | Linux: pauses ffmpeg. Windows: not available in current implementation. |
| **Resume** | Paused | Linux: resumes ffmpeg. Windows: not available in current implementation. |
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
recording_YYYY-MM-DD_HH-MM-SS.mp3
```

On Windows, the default resolves to `C:\Users\<you>\recordings`.

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
   --out recordings/summaries/meeting-summary.txt
```

### From an audio file (transcribe + diarize + summarize)

```bash
python scripts/summarize_transcript.py \
   --audio recordings/recording_2026-03-13_15-00-00.mp3 \
   --min-speakers 1 \
   --max-speakers 3 \
   --out recordings/summaries/meeting-summary.txt
```

### Custom model

```bash
python scripts/summarize_transcript.py \
   --transcript <file> \
   --model <your-ollama-model-tag> \
   --out recordings/summaries/meeting-summary.txt
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
| PowerShell blocks venv activation | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then `.\venv\Scripts\Activate.ps1` |
| WhisperX import error | Run `python -m pip install -r requirements.txt` inside the active venv |
| Hugging Face cache `WinError 1314` on Windows | Enable Developer Mode in Windows or run elevated once; then retry summary |
