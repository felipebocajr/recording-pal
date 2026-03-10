"""
bot.py – Discord bot for speaker-diarized meeting transcription.

The bot listens for the ``!transcribe`` command with an attached audio file,
runs the WhisperX + pyannote pipeline, and replies with the transcript.
It also exposes slash commands ``/record`` and ``/stop`` to capture voice chat
audio into a local ``.wav`` file.

Environment variables (loaded from ``.env`` via python-dotenv):
    DISCORD_BOT_TOKEN  – Discord bot token (required)
    HF_TOKEN           – Hugging Face access token (required)
    WHISPER_MODEL      – WhisperX model size (default: "base")
    DEVICE             – "cpu" or "cuda" (default: "cpu")
    COMPUTE_TYPE       – quantisation type (default: "auto")
    BATCH_SIZE         – WhisperX batch size (default: 16)
    RECORDINGS_DIR     – output folder for ``/stop`` recordings (default: "recordings")
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import threading
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import discord
from discord.ext import commands

logger = logging.getLogger(__name__)

# Audio extensions the bot will accept
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}

# Discord message character limit
DISCORD_MSG_LIMIT = 2000
DEFAULT_UPLOAD_LIMIT_BYTES = 8 * 1024 * 1024


@dataclass
class RecordingSession:
    """In-memory state for one guild voice recording session."""

    sink: "PCMRecordingSink"
    voice_client: discord.VoiceProtocol
    channel_name: str
    is_stopping: bool = False


class PCMRecordingSink:
    """Collect raw PCM frames from voice-recv and export them as a WAV file."""

    def __init__(self) -> None:
        try:
            from discord.ext import voice_recv
        except ImportError as exc:
            raise RuntimeError(
                "discord-ext-voice-recv is required for /record. Install with: "
                "pip install discord-ext-voice-recv"
            ) from exc

        class _Sink(voice_recv.AudioSink):
            def __init__(self, owner: "PCMRecordingSink") -> None:
                super().__init__()
                self._owner = owner

            def wants_opus(self) -> bool:
                return False

            def write(self, user: discord.Member | None, data: object) -> None:
                pcm = getattr(data, "pcm", b"")
                if pcm:
                    self._owner.append_pcm(pcm)

        self._sink = _Sink(self)
        self._lock = threading.Lock()
        self._frames = bytearray()

    @property
    def audio_sink(self) -> object:
        return self._sink

    def append_pcm(self, chunk: bytes) -> None:
        with self._lock:
            self._frames.extend(chunk)

    def write_wav(self, output_path: Path) -> int:
        with self._lock:
            frames = bytes(self._frames)

        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(48_000)
            wav_file.writeframes(frames)

        return len(frames)


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
    recording_sessions: dict[int, RecordingSession] = {}
    state_lock = asyncio.Lock()
    tree_synced = False

    @bot.event
    async def on_ready() -> None:
        nonlocal tree_synced
        logger.info("Bot is online as %s (id=%s)", bot.user, bot.user.id)
        async with state_lock:
            if not tree_synced:
                synced = await bot.tree.sync()
                tree_synced = True
                logger.info("Synced %d slash command(s).", len(synced))

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

    @bot.tree.command(name="record", description="Start recording the voice channel you are currently in.")
    async def record(interaction: discord.Interaction) -> None:
        if interaction.guild is None or interaction.user is None:
            await interaction.response.send_message("❌ This command must be used in a server.", ephemeral=True)
            return

        voice_state = getattr(interaction.user, "voice", None)
        if voice_state is None or voice_state.channel is None:
            await interaction.response.send_message(
                "❌ Join a voice channel first, then run `/record`.",
                ephemeral=True,
            )
            return

        try:
            from discord.ext import voice_recv
        except ImportError:
            await interaction.response.send_message(
                "❌ Voice recording extension is missing. Install `discord-ext-voice-recv` and restart the bot.",
                ephemeral=True,
            )
            return

        channel = voice_state.channel
        try:
            voice_client = await channel.connect(cls=voice_recv.VoiceRecvClient)
            sink = PCMRecordingSink()
            voice_client.listen(sink.audio_sink)
        except Exception as exc:
            logger.exception("Failed to start voice recording")
            await interaction.response.send_message(f"❌ Failed to start recording: {exc}", ephemeral=True)
            return

        async with state_lock:
            if interaction.guild.id in recording_sessions:
                await voice_client.disconnect(force=True)
                await interaction.response.send_message(
                    "⚠️ A recording is already in progress for this server. Use `/stop` first.",
                    ephemeral=True,
                )
                return
            recording_sessions[interaction.guild.id] = RecordingSession(
                sink=sink,
                voice_client=voice_client,
                channel_name=channel.name,
            )
        await interaction.response.send_message(
            f"🔴 Recording started in **{channel.name}**.\nUse `/stop` to save the audio file.",
            ephemeral=False,
        )

    @bot.tree.command(name="stop", description="Stop active recording and save it to the recordings folder.")
    async def stop(interaction: discord.Interaction) -> None:
        if interaction.guild is None:
            await interaction.response.send_message("❌ This command must be used in a server.", ephemeral=True)
            return

        async with state_lock:
            session = recording_sessions.get(interaction.guild.id)
            if session is None:
                await interaction.response.send_message(
                    "❌ No active recording found. Use `/record` first.",
                    ephemeral=True,
                )
                return
            if session.is_stopping:
                await interaction.response.send_message(
                    "⏳ Recording is already being stopped. Please wait.",
                    ephemeral=True,
                )
                return
            session.is_stopping = True

        await interaction.response.defer(thinking=True)
        try:
            try:
                session.voice_client.stop_listening()
            except Exception:
                logger.debug(
                    "stop_listening failed or was already stopped (guild_id=%s)",
                    interaction.guild.id,
                    exc_info=True,
                )

            try:
                await session.voice_client.disconnect(force=True)
            except Exception:
                logger.debug("Voice disconnect failed", exc_info=True)

            recordings_dir = Path(_get_env("RECORDINGS_DIR", "recordings"))
            recordings_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output_path = recordings_dir / f"meeting-{interaction.guild.id}-{timestamp}.wav"
            session.sink.write_wav(output_path)
            file_size = output_path.stat().st_size
            upload_limit = getattr(interaction.guild, "filesize_limit", DEFAULT_UPLOAD_LIMIT_BYTES)

            if file_size <= upload_limit:
                await interaction.followup.send(
                    f"✅ Recording stopped from **{session.channel_name}**.\n"
                    f"Saved `{output_path}` ({file_size} bytes).",
                    file=discord.File(str(output_path), filename=output_path.name),
                )
            else:
                await interaction.followup.send(
                    f"✅ Recording stopped from **{session.channel_name}**.\n"
                    f"Saved `{output_path}` ({file_size} bytes), but it exceeds Discord upload limit "
                    f"({file_size} > {upload_limit} bytes), so only the local file was kept."
                )
        finally:
            async with state_lock:
                recording_sessions.pop(interaction.guild.id, None)

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
