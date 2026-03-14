"""
summarize.py – Local LLM meeting summarization from transcript files.

Uses Ollama runtime.
Default target model:
    hf.co/unsloth/Qwen3.5-9B-GGUF:Q4_K_M  (fallback: qwen3.5:4b)
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PULL_TIMEOUT_SECONDS = 60 * 30
DEFAULT_RUN_TIMEOUT_SECONDS = 60 * 30
DEFAULT_FALLBACK_MODEL = "qwen3.5:4b"


def _load_transcript_text(path: Path) -> str:
    suffix = path.suffix.lower()
    content = path.read_text(encoding="utf-8").strip()

    if not content:
        return ""

    if suffix != ".json":
        return content

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return content

    if not isinstance(data, list):
        return content

    lines: list[str] = []
    for index, turn in enumerate(data, start=1):
        if not isinstance(turn, dict):
            continue
        speaker = str(turn.get("speaker", "UNKNOWN"))
        text = str(turn.get("text", "")).strip()
        start = turn.get("start")
        if not text:
            continue

        if isinstance(start, (int, float)):
            sec = int(start)
            h, rem = divmod(sec, 3600)
            m, s = divmod(rem, 60)
            ts = f"{h:02d}:{m:02d}:{s:02d}"
        else:
            ts = "??:??:??"

        lines.append(f"{index} [{speaker} @ {ts}]")
        lines.append(text)
        lines.append("")

    return "\n".join(lines).strip()


def _build_prompt(transcript: str, language: str = "pt-BR") -> str:
    if language.lower().startswith("pt"):
        instruction = (
            "Você é um assistente de reuniões. Gere um resumo fiel ao transcript. "
            "Não invente informações. Se algo não estiver claro, marque como 'não informado'.\n\n"
            "Regra crítica: responda APENAS com o conteúdo final do resumo. "
            "NÃO inclua thinking process, raciocínio, análise interna, meta-comentários, "
            "ou explicações sobre como você chegou na resposta.\n\n"
            "Formato obrigatório:\n"
            "1) Resumo executivo (5 bullets)\n"
            "2) Decisões tomadas\n"
            "3) Itens de ação (Responsável | Ação | Prazo)\n"
            "4) Perguntas em aberto\n"
            "5) Riscos e bloqueios\n"
            "6) Próximos passos\n"
        )
    else:
        instruction = (
            "You are a meeting assistant. Produce a faithful summary from the transcript. "
            "Do not invent details. If unknown, mark as 'not specified'.\n\n"
            "Critical rule: output ONLY the final summary. "
            "DO NOT include thinking process, chain-of-thought, internal analysis, "
            "meta commentary, or reasoning notes.\n\n"
            "Required format:\n"
            "1) Executive summary (5 bullets)\n"
            "2) Decisions made\n"
            "3) Action items (Owner | Task | Deadline)\n"
            "4) Open questions\n"
            "5) Risks and blockers\n"
            "6) Next steps\n"
        )

    return (
        f"{instruction}\n"
        "TRANSCRIPT START\n"
        f"{transcript}\n"
        "TRANSCRIPT END\n"
    )


def _sanitize_summary_output(raw_text: str, language: str = "pt-BR") -> str:
    text = (raw_text or "").strip()
    if not text:
        return ""

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)

    lower = text.lower()
    done_marker = "done thinking"
    done_idx = lower.find(done_marker)
    if done_idx != -1:
        text = text[done_idx + len(done_marker):].lstrip(" .:\n\t-")

    start_markers_pt = [
        "1) resumo executivo",
        "**1) resumo executivo",
        "*   **1) resumo executivo",
    ]
    start_markers_en = [
        "1) executive summary",
        "**1) executive summary",
        "*   **1) executive summary",
    ]
    start_markers = start_markers_pt if language.lower().startswith("pt") else start_markers_en

    lower = text.lower()
    starts = [lower.find(marker) for marker in start_markers if lower.find(marker) != -1]
    if starts:
        text = text[min(starts):].strip()

    split_markers = [
        "\n---\n",
        "\nI've provided",
        "\nI have provided",
        "\nNow I'm ready",
    ]
    for marker in split_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].rstrip()

    return text.strip()


def _check_ollama_ready() -> None:
    try:
        subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Ollama não está instalado (ou não está no PATH). "
            "Instale em: https://ollama.com/download"
        ) from exc
    except subprocess.SubprocessError as exc:
        raise RuntimeError(f"Falha ao validar instalação do Ollama: {exc}") from exc

    try:
        subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(
            "Ollama parece não estar em execução. "
            "Inicie com `ollama serve` em outro terminal. "
            f"Detalhe: {stderr or exc}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Timeout ao conectar no Ollama. Verifique se o daemon está ativo (`ollama serve`)."
        ) from exc


def _ensure_model_available(model: str, auto_pull: bool, pull_timeout_seconds: int) -> None:
    try:
        subprocess.run(
            ["ollama", "show", model],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return
    except subprocess.CalledProcessError:
        if not auto_pull:
            raise RuntimeError(
                f"Modelo não encontrado no Ollama: {model}. "
                f"Baixe manualmente com: `ollama pull {model}`"
            )

    logger.info("Model not found locally. Pulling with Ollama: %s", model)
    try:
        subprocess.run(
            ["ollama", "pull", model],
            check=True,
            text=True,
            timeout=pull_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Timeout durante download do modelo via Ollama. "
            f"Tente novamente ou rode manualmente: `ollama pull {model}`"
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"Falha ao baixar modelo via Ollama: {stderr or exc}") from exc


def _run_model_once(
    model: str,
    prompt: str,
    run_timeout_seconds: int,
    allow_cpu_fallback: bool,
) -> str:
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            capture_output=True,
            check=True,
            timeout=run_timeout_seconds,
        )
        return (result.stdout or "").strip()
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Timeout durante geração do resumo com Ollama. "
            "Seu modelo pode ser grande ou a máquina pode estar sobrecarregada. "
            "Aumente `--run-timeout` e tente novamente."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        lower_stderr = stderr.lower()
        gpu_oom = (
            "unable to allocate cuda" in lower_stderr
            or "cuda" in lower_stderr and "failed to load model" in lower_stderr
            or "out of memory" in lower_stderr and "cuda" in lower_stderr
        )

        if allow_cpu_fallback and gpu_oom:
            logger.warning(
                "Ollama failed on GPU allocation for %s. Retrying on CPU (slower).",
                model,
            )
            cpu_env = dict(os.environ)
            cpu_env["OLLAMA_NUM_GPU"] = "0"
            try:
                cpu_result = subprocess.run(
                    ["ollama", "run", model],
                    input=prompt,
                    text=True,
                    capture_output=True,
                    check=True,
                    timeout=run_timeout_seconds,
                    env=cpu_env,
                )
                return (cpu_result.stdout or "").strip()
            except subprocess.CalledProcessError as cpu_exc:
                cpu_stderr = (cpu_exc.stderr or "").strip()
                raise RuntimeError(
                    "Ollama falhou na GPU e também na execução em CPU. "
                    f"Erro CPU: {cpu_stderr or cpu_exc}"
                ) from cpu_exc
            except subprocess.TimeoutExpired as cpu_timeout_exc:
                raise RuntimeError(
                    "Timeout no fallback CPU do Ollama. "
                    "Aumente `--run-timeout` e tente novamente."
                ) from cpu_timeout_exc

        raise RuntimeError(f"Ollama command failed: {stderr or exc}") from exc


def summarize_transcript_with_ollama(
    transcript_path: str,
    output_path: str,
    *,
    model: str = "hf.co/unsloth/Qwen3.5-9B-GGUF:Q4_K_M",
    fallback_model: str | None = DEFAULT_FALLBACK_MODEL,
    language: str = "pt-BR",
    auto_pull: bool = True,
    pull_timeout_seconds: int = DEFAULT_PULL_TIMEOUT_SECONDS,
    run_timeout_seconds: int = DEFAULT_RUN_TIMEOUT_SECONDS,
    allow_cpu_fallback: bool = True,
) -> Path:
    """Generate a local meeting summary from transcript using Ollama."""
    transcript_file = Path(transcript_path).expanduser().resolve()
    if not transcript_file.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_file}")

    transcript = _load_transcript_text(transcript_file)
    if not transcript:
        raise RuntimeError("Transcript file is empty.")

    _check_ollama_ready()
    prompt = _build_prompt(transcript, language=language)

    models_to_try: list[str] = [model]
    if fallback_model and fallback_model != model:
        models_to_try.append(fallback_model)

    errors: list[str] = []
    text = ""
    for index, selected_model in enumerate(models_to_try, start=1):
        logger.info("Running Ollama model (%d/%d): %s", index, len(models_to_try), selected_model)
        try:
            _ensure_model_available(
                selected_model,
                auto_pull=auto_pull,
                pull_timeout_seconds=pull_timeout_seconds,
            )
            text = _run_model_once(
                model=selected_model,
                prompt=prompt,
                run_timeout_seconds=run_timeout_seconds,
                allow_cpu_fallback=allow_cpu_fallback,
            )
            if text:
                break
            errors.append(f"{selected_model}: returned empty response")
        except RuntimeError as exc:
            logger.warning("Model failed (%s): %s", selected_model, exc)
            errors.append(f"{selected_model}: {exc}")

    if not text:
        detail = " | ".join(errors) if errors else "unknown error"
        raise RuntimeError(f"All configured models failed. Details: {detail}")

    text = _sanitize_summary_output(text, language=language)
    if not text:
        raise RuntimeError(
            "Model returned only reasoning/meta output after sanitization. "
            "Try another model or rerun with fallback enabled."
        )

    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    logger.info("Summary saved -> %s", out)
    return out
