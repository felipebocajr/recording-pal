#!/usr/bin/env python3
"""
run_discord_bot.py – Start the Transcriptor Discord bot.

Usage
-----
    python scripts/run_discord_bot.py

Make sure a ``.env`` file exists in the project root (copy from
``.env.example``) with at least ``DISCORD_BOT_TOKEN`` and ``HF_TOKEN`` set.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from transcriptor_bot.bot import run_bot  # noqa: E402

if __name__ == "__main__":
    run_bot()
