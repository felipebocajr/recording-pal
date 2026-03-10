"""
bot.py – Discord bot for speaker-diarized meeting transcription.

The bot listens for the ``!transcribe`` command with an attached audio file,
runs the WhisperX + pyannote pipeline, and replies with the transcript.

Environment variables (loaded from ``.env`` via python-dotenv):
    DISCORD_BOT_TOKEN  – Discord bot token (required)
    HF_TOKEN           – Hugging Face access token (required)
    WHISPER_MODEL      – WhisperX model size (default: "base")
    DEVICE             – "cpu" or "cuda" (default: "cpu")
    COMPUTE_TYPE       – quantisation type (default: "auto")
    BATCH_SIZE         – WhisperX batch size (default: 16)
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

import discord
from discord.ext import commands

logger = logging.getLogger(__name__)

# Audio extensions the bot will accept
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}

# Discord message character limit
DISCORD_MSG_LIMIT = 2000


def _get_env(name: str, default: str | None = None, required: bool = False) -> str:
    """Return an environment variable or *default*.  Raise if *required* and missing."""
    value = os.environ.get(name, default or "")
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def create_bot() -> commands.Bot:
    """Build and return the configured :class:`~commands.Bot` instance."""

    intents = discord.Intents.default()
    intents.message_content = True

    bot = commands.Bot(command_prefix="!", intents=intents)

    @bot.event
    async def on_ready() -> None:
        logger.info("Bot is online as %s (id=%s)", bot.user, bot.user.id)

    @bot.command(name="transcribe", help="Transcribe an attached audio file with speaker diarization.")
    async def transcribe(ctx: commands.Context, min_speakers: int = 2, max_speakers: int = 3) -> None:
        """Download the attached audio, run the pipeline, and reply with the transcript.

        Usage::

            !transcribe                     ← uses defaults (min=2, max=3)
            !transcribe 2 4                 ← min_speakers=2, max_speakers=4
        """
        # --- validate attachment ------------------------------------------------
        if not ctx.message.attachments:
            await ctx.reply(
                "❌ Please attach an audio file to your message.\n"
                "**Usage:** `!transcribe [min_speakers] [max_speakers]`\n"
                "**Example:** `!transcribe 2 4`\n"
                f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
            return

        attachment = ctx.message.attachments[0]
        ext = Path(attachment.filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            await ctx.reply(
                f"❌ Unsupported file type `{ext}`.\n"
                f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
            return

        # --- set up temp directory ----------------------------------------------
        tmp_dir = tempfile.mkdtemp(prefix="transcriptor_")
        audio_path = os.path.join(tmp_dir, attachment.filename)
        out_dir = os.path.join(tmp_dir, "outputs")
        os.makedirs(out_dir, exist_ok=True)

        try:
            # --- download -------------------------------------------------------
            status_msg = await ctx.reply("⏳ **Downloading audio…**")
            await attachment.save(audio_path)

            # --- read config from env -------------------------------------------
            hf_token = _get_env("HF_TOKEN", required=True)
            model_size = _get_env("WHISPER_MODEL", "base")
            device = _get_env("DEVICE", "cpu")
            compute_type = _get_env("COMPUTE_TYPE", "auto")
            batch_size = int(_get_env("BATCH_SIZE", "16"))

            # --- Step 1: Transcription ------------------------------------------
            await status_msg.edit(content="⏳ **Step 1/3 – Transcribing audio with WhisperX…**")
            from transcriptor_bot.transcribe import run_whisperx_transcribe

            whisperx_result = run_whisperx_transcribe(
                audio_path=audio_path,
                model_size=model_size,
                device=device,
                compute_type=compute_type,
                batch_size=batch_size,
            )

            # --- Step 2: Diarization --------------------------------------------
            await status_msg.edit(content="⏳ **Step 2/3 – Running speaker diarization…**")
            from transcriptor_bot.diarize import run_pyannote_diarization

            diarization, rttm_lines = run_pyannote_diarization(
                audio_path=audio_path,
                hf_token=hf_token,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

            # --- Step 3: Assign + Render ----------------------------------------
            await status_msg.edit(content="⏳ **Step 3/3 – Assigning words to speakers…**")
            from transcriptor_bot.assign import (
                assign_words_to_speakers,
                build_speaker_turns,
                render_transcript,
                turns_to_json,
            )

            assigned_words = assign_words_to_speakers(
                word_segments=whisperx_result["word_segments"],
                diarization=diarization,
            )
            turns = build_speaker_turns(assigned_words)
            transcript_text = render_transcript(turns)
            transcript_json = turns_to_json(turns)

            # --- Save outputs locally -------------------------------------------
            txt_path = os.path.join(out_dir, "speaker_transcript.txt")
            json_path = os.path.join(out_dir, "speaker_transcript.json")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(transcript_text)
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(transcript_json)

            # --- Reply with transcript ------------------------------------------
            await status_msg.edit(content="✅ **Transcription complete!**")

            # If short enough, send as a message; otherwise attach as files
            if len(transcript_text) <= DISCORD_MSG_LIMIT:
                await ctx.reply(f"```\n{transcript_text}\n```")
            else:
                await ctx.reply(
                    "📄 The transcript is too long for a message — see attached files.",
                    files=[
                        discord.File(txt_path, filename="speaker_transcript.txt"),
                        discord.File(json_path, filename="speaker_transcript.json"),
                    ],
                )

            # Always attach the files for convenience
            if len(transcript_text) <= DISCORD_MSG_LIMIT:
                await ctx.send(
                    files=[
                        discord.File(txt_path, filename="speaker_transcript.txt"),
                        discord.File(json_path, filename="speaker_transcript.json"),
                    ],
                )

        except RuntimeError as exc:
            logger.exception("Pipeline error")
            await ctx.reply(f"❌ **Error:** {exc}")
        except Exception as exc:
            logger.exception("Unexpected error during transcription")
            await ctx.reply(f"❌ **Unexpected error:** {exc}")
        finally:
            # --- cleanup temp files ---------------------------------------------
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return bot


def run_bot() -> None:
    """Load environment, create the bot, and start the event loop."""
    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    token = _get_env("DISCORD_BOT_TOKEN", required=True)
    bot = create_bot()
    bot.run(token)
