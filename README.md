# Transcriptor-bot

A toolkit for **speaker-diarized meeting transcription** using
[WhisperX](https://github.com/m-bain/whisperx) (ASR + word alignment) and
[pyannote.audio](https://github.com/pyannote/pyannote-audio) (speaker diarization).

Works as a **Discord bot**, a **CLI tool**, a **Python library**, or a **Google Colab notebook**.

Given a **single mixed audio file** from a meeting (multiple speakers), the
pipeline produces:

| Artifact | Description |
|---|---|
| `whisperx_result.json` | Full transcription segments + word-level timestamps |
| `diarization.rttm` | Speaker turns in RTTM format |
| `speaker_transcript.txt` | Human-readable, speaker-labelled transcript |
| `speaker_transcript.json` | Machine-readable list of speaker turns |

---

## 🤖 Discord Bot Setup (Step by Step)

The fastest way to use Transcriptor-bot is through Discord. Follow these steps:

### 1. Create a Discord Bot

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications).
2. Click **New Application** → give it a name (e.g. "Transcriptor Bot") → **Create**.
3. In the left sidebar, click **Bot**.
4. Click **Reset Token** → copy the token. **Save it — you'll need it soon.**
5. Scroll down and enable **Message Content Intent** (required to read commands).

### 2. Invite the Bot to Your Server

1. In the Developer Portal, go to **OAuth2 → URL Generator**.
2. Under **Scopes**, check `bot`.
3. Under **Bot Permissions**, check:
   - `Send Messages`
   - `Attach Files`
   - `Read Message History`
   - `Connect`
   - `Speak`
   - `Use Voice Activity`
4. Copy the generated URL at the bottom and open it in your browser.
5. Select your server and click **Authorize**.

### 3. Get a Hugging Face Token

1. Create a free account at <https://huggingface.co/join>.
2. Generate an access token at <https://huggingface.co/settings/tokens> (type: **Read**).
3. Accept the licence for **both** models (one-time, free):
   - <https://huggingface.co/pyannote/speaker-diarization-3.1>
   - <https://huggingface.co/pyannote/segmentation-3.0>

### 4. Install & Configure

```bash
# Clone the repo
git clone https://github.com/felipebocajr/Transcriptor-bot.git
cd Transcriptor-bot

# Install system dependency
#   macOS:  brew install ffmpeg
#   Ubuntu: sudo apt-get install ffmpeg

# Install Python dependencies
pip install -r requirements.txt
pip install git+https://github.com/m-bain/whisperx.git

# For GPU (CUDA 11.8 example):
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
# For CPU only:
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Create your .env file from the template
cp .env.example .env
```

Now edit `.env` and fill in your tokens:

```env
DISCORD_BOT_TOKEN=paste_your_discord_bot_token_here
HF_TOKEN=hf_paste_your_huggingface_token_here
RECORDINGS_DIR=recordings
```

### 5. Run the Bot

```bash
python scripts/run_discord_bot.py
```

You should see a log message confirming the bot is online.

### 6. Use the Bot in Discord

Upload an audio file to any channel the bot can see and type:

```
!transcribe
```

Or specify speaker counts:

```
!transcribe 2 5
```

The bot will:
1. Download the attached audio file.
2. Transcribe it with WhisperX.
3. Run speaker diarization with pyannote.
4. Reply with the full speaker-labelled transcript and attach output files.

**Supported audio formats:** `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.webm`, `.mp4`

To record a live Discord meeting audio into a local/Colab folder:

1. Join a voice channel.
2. Run:

```
/record
```

3. When the meeting ends, run:

```
/stop
```

The bot saves a `.wav` file in `RECORDINGS_DIR` (default: `recordings/`) and also uploads that file in the channel.

---

## Quick Start (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/felipebocajr/Transcriptor-bot/blob/main/notebooks/colab_whisperx_pyannote_diarization.ipynb)

1. Click the badge above to open the notebook in Colab.
2. Select **Runtime → Change runtime type → T4 GPU** for best performance.
3. Follow the steps in the notebook:
   - Install dependencies (ffmpeg + Python packages).
   - Set your Hugging Face token (see [Getting an HF Token](#3-get-a-hugging-face-token)).
   - Upload or point to your audio file.
   - Set `MIN_SPEAKERS` / `MAX_SPEAKERS` (default: 3).
   - Run all cells — outputs appear in `/content/outputs/`.
   - Download the zipped results.

---

## Prerequisites

### Getting a Hugging Face Token

pyannote.audio models are **gated** (free, but require consent):

1. Create a free account at <https://huggingface.co/join>.
2. Generate an access token at <https://huggingface.co/settings/tokens>
   (type: **Read**).
3. Accept the licence for **both** models (one-time):
   - <https://huggingface.co/pyannote/speaker-diarization-3.1>
   - <https://huggingface.co/pyannote/segmentation-3.0>

### Setting `HF_TOKEN`

The token can be provided in several ways (the code tries them in order):

| Method | How |
|---|---|
| **`.env` file** (recommended for the bot) | Add `HF_TOKEN=hf_…` to your `.env` file |
| **Colab Secrets** (recommended for Colab) | Click the 🔑 icon in the left sidebar, add secret `HF_TOKEN`, enable notebook access |
| **Environment variable** | `export HF_TOKEN=hf_…` before running the script |
| **CLI flag** | `--hf-token hf_…` when calling `diarize_meeting.py` |

---

## Local / CLI Usage

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/felipebocajr/Transcriptor-bot.git
cd Transcriptor-bot

# 2. Install system dependency (macOS/Linux)
#    macOS:  brew install ffmpeg
#    Ubuntu: sudo apt-get install ffmpeg

# 3. Install Python deps
pip install git+https://github.com/m-bain/whisperx.git
pip install pyannote.audio huggingface-hub

# For GPU (CUDA 11.8 example):
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
# For CPU only:
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Running the CLI

```bash
export HF_TOKEN=hf_your_token_here

python scripts/diarize_meeting.py \
    --audio  path/to/meeting.wav \
    --min-speakers 3 \
    --max-speakers 3 \
    --out    outputs/
```

All four output files will be written to `outputs/`.

#### Full CLI options

```
usage: diarize_meeting.py [-h] --audio PATH [--hf-token TOKEN]
                          [--min-speakers N] [--max-speakers N]
                          [--out DIR] [--model MODEL] [--device {cpu,cuda}]
                          [--compute-type {auto,float16,int8,float32}]
                          [--language LANG] [--batch-size N] [-v]

options:
  --audio PATH          Path to the input audio file (wav/mp3/m4a/…)
  --hf-token TOKEN      Hugging Face access token (or set HF_TOKEN env var)
  --min-speakers N      Minimum number of speakers (default: 3)
  --max-speakers N      Maximum number of speakers (default: 3)
  --out DIR             Output directory (default: outputs/)
  --model MODEL         WhisperX model size (default: base)
  --device {cpu,cuda}   Device (default: cpu)
  --compute-type …      Quantisation type (default: auto)
  --language LANG       Language code, e.g. "en" (default: auto-detect)
  --batch-size N        WhisperX batch size (default: 16)
  -v, --verbose         Enable verbose logging
```

---

## Python API

```python
from transcriptor_bot.transcribe import run_whisperx_transcribe
from transcriptor_bot.diarize    import run_pyannote_diarization
from transcriptor_bot.assign     import (
    assign_words_to_speakers,
    build_speaker_turns,
    render_transcript,
)

# 1. Transcribe + align
result = run_whisperx_transcribe("meeting.wav", model_size="base", device="cpu")

# 2. Diarize
diarization, rttm_lines = run_pyannote_diarization(
    "meeting.wav", hf_token="hf_…", min_speakers=3, max_speakers=3
)

# 3. Assign & render
words   = assign_words_to_speakers(result["word_segments"], diarization)
turns   = build_speaker_turns(words)
print(render_transcript(turns))
```

---

## GPU vs CPU — Expected Runtime

| Setup | Model | ~60 min audio |
|---|---|---|
| Colab T4 GPU | `large-v2` | ~8–12 min |
| Colab T4 GPU | `base` | ~2–3 min |
| CPU only | `base` | ~20–40 min |
| CPU only | `tiny` | ~8–15 min |

> **Tip:** The diarization step (pyannote) takes a roughly fixed ~1–3 minutes
> per hour of audio regardless of CPU/GPU.

---

## Repository Structure

```
Transcriptor-bot/
├── notebooks/
│   └── colab_whisperx_pyannote_diarization.ipynb  ← Colab notebook
├── transcriptor_bot/                               ← Python package
│   ├── __init__.py
│   ├── transcribe.py   – WhisperX ASR + alignment
│   ├── diarize.py      – pyannote diarization
│   ├── assign.py       – word→speaker assignment + rendering
│   └── bot.py          – Discord bot integration
├── scripts/
│   ├── diarize_meeting.py  ← CLI entry point
│   └── run_discord_bot.py  ← Discord bot entry point
├── outputs/                ← generated artefacts (git-ignored)
├── .env.example            ← template for environment variables
├── requirements.txt
└── README.md
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `Missing required environment variable: DISCORD_BOT_TOKEN` | Add your Discord bot token to `.env` |
| `Missing required environment variable: HF_TOKEN` | Add your HF token to `.env` |
| `Privileged intent … not enabled` | Enable **Message Content Intent** in the [Developer Portal](https://discord.com/developers/applications) → Bot |
| `RuntimeError: No Hugging Face token found` | Set `HF_TOKEN` env var or use Colab Secrets |
| `pyannote model access denied (403/401)` | Accept both model licences on the Hub (links above) |
| CUDA out of memory | Reduce `BATCH_SIZE` in `.env` or switch to a smaller `WHISPER_MODEL` |
| `ffmpeg not found` | `sudo apt-get install ffmpeg` (Linux) or `brew install ffmpeg` (macOS) |
| WhisperX import error | Run `pip install git+https://github.com/m-bain/whisperx.git` |
| Bot is online but doesn't respond | Make sure the bot has permission to read messages in the channel |
