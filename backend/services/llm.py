"""LLM provider layer for RegCheck comparisons.

Everything that talks to a model provider lives here: client construction
(OpenAI / DeepSeek / Groq / Claude), per-provider model resolution, the chat
helpers used by the comparison flows, provider-error classification, and the
response-parsing utilities. Clients are built lazily so the app/worker boot even
when a given provider's key is unset — an unset key fails only that provider's
path, not the whole process.

This module is intentionally free of comparison/business logic; ``comparisons``
imports these helpers and orchestrates them.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

from groq import Groq
from openai import OpenAI

try:  # Optional provider SDK; only required when client_choice == "claude".
    from anthropic import Anthropic
except ImportError:  # pragma: no cover - dependency optional at runtime
    Anthropic = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_MODEL = "gpt-5"
DEFAULT_GPT_OSS_MODEL = "openai/gpt-oss-120b"
DEFAULT_DEEPSEEK_MODEL = "deepseek-reasoner"
DEFAULT_GROQ_MODEL = "qwen/qwen3.6-27b"
DEFAULT_CLAUDE_MODEL = "claude-opus-4-8"

# Client choices that run through the OpenAI SDK. (GPT-OSS is an open-weight
# model that OpenAI's hosted API does not serve — it is routed via Groq below.)
_OPENAI_CLIENTS = {"openai"}


def _env_str(name: str, default: str | None = None) -> str:
    value = (os.environ.get(name) or "").strip()
    if value:
        return value
    if default is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


# ---- provider error classification ------------------------------------------

def _is_provider_auth_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 401:
        return True
    exc_name = type(exc).__name__.lower()
    text = str(exc).lower()
    return (
        "authentication" in exc_name
        or "unauthorized" in text
        or "invalid_api_key" in text
        or "invalid api key" in text
    )


def _raise_provider_auth_error(provider: str, env_var: str, exc: Exception) -> None:
    raise RuntimeError(
        f"{provider} authentication failed. Check Heroku config var {env_var} "
        f"or choose a different model provider."
    ) from exc


def _is_response_format_error(exc: Exception) -> bool:
    text = str(exc).lower()
    # Includes Groq's `json_validate_failed` (400) — some models, especially
    # reasoning ones, can't reliably satisfy strict JSON mode, so we retry
    # without response_format and extract the JSON from the free-form reply.
    return (
        "response_format" in text
        or "json_object" in text
        or "json mode" in text
        or "json_validate_failed" in text
        or "failed to validate json" in text
    )


# ---- per-provider model resolution -------------------------------------------

def _openai_model() -> str:
    return _env_str("OPENAI_COMPARISON_MODEL", _env_str("OPENAI_MODEL", DEFAULT_OPENAI_MODEL))


def _gpt_oss_model() -> str:
    return _env_str("GPT_OSS_MODEL", DEFAULT_GPT_OSS_MODEL)


def _openai_experiment_model() -> str:
    return _env_str("OPENAI_EXPERIMENT_MODEL", _openai_model())


def _openai_family_model(client_choice: str, *, experiment: bool = False) -> str:
    return _openai_experiment_model() if experiment else _openai_model()


def _deepseek_model() -> str:
    return _env_str("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)


def _groq_model() -> str:
    return _env_str("GROQ_MODEL", DEFAULT_GROQ_MODEL)


def _claude_model() -> str:
    return _env_str("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)


def _claude_max_tokens(default: int = 16000) -> int:
    return _env_int("CLAUDE_MAX_TOKENS", default)


def _normalize_reasoning_effort_value(value: str | None) -> str | None:
    normalized = (value or "").strip().lower()
    if not normalized:
        return None
    if normalized not in {"low", "medium", "high"}:
        return "medium"
    return normalized


# ---- response parsing --------------------------------------------------------

def _message_content_to_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for element in content:
            if hasattr(element, "text"):
                part_type = (getattr(element, "type", "") or "").lower()
                if part_type in {"reasoning", "thinking", "tool_calls"}:
                    continue
                text_value = getattr(element, "text", None)
                if text_value:
                    parts.append(str(text_value))
                continue
            if isinstance(element, dict):
                part_type = (element.get("type") or "").lower()
                if part_type in {"reasoning", "thinking", "tool_calls"}:
                    continue
                text_value = element.get("text")
                if text_value:
                    parts.append(str(text_value))
            elif isinstance(element, str):
                parts.append(element)
        return "".join(parts).strip()
    if content is None:
        return ""
    return str(content)


def _strip_deepseek_reasoning(content: str) -> str:
    if not content:
        return content
    closing_tag = "</think>"
    closing_index = content.find(closing_tag)
    if closing_index != -1:
        content = content[closing_index + len(closing_tag) :]
    return content.lstrip()


def _extract_json_payload(raw_text: str) -> str:
    if not raw_text:
        return raw_text
    text = raw_text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    return text.strip()


# ---- OpenAI-family chat helpers ----------------------------------------------

def _openai_error_param(exc: Exception) -> str | None:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            param = error.get("param")
            if isinstance(param, str) and param.strip():
                return param.strip()

    message = str(exc)
    for candidate in ("reasoning_effort", "response_format"):
        if candidate in message:
            return candidate
    return None


def _openai_chat_json(
    openai_client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    reasoning_effort: str | None = None,
) -> str:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    normalized_effort = _normalize_reasoning_effort_value(reasoning_effort)
    if normalized_effort:
        kwargs["reasoning_effort"] = normalized_effort

    while True:
        try:
            response = openai_client.chat.completions.create(**kwargs)
            break
        except Exception as exc:
            param = _openai_error_param(exc)
            if param and param in kwargs:
                logger.info(
                    "OpenAI call failed with %s; retrying without it",
                    param,
                    extra={"model": model},
                    exc_info=exc,
                )
                kwargs.pop(param, None)
                continue
            raise
    return _message_content_to_text(response.choices[0].message)


def _openai_chat_text(
    openai_client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    reasoning_effort: str | None = None,
) -> str:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    normalized_effort = _normalize_reasoning_effort_value(reasoning_effort)
    if normalized_effort:
        kwargs["reasoning_effort"] = normalized_effort

    while True:
        try:
            response = openai_client.chat.completions.create(**kwargs)
            break
        except Exception as exc:
            param = _openai_error_param(exc)
            if param and param in kwargs:
                logger.info(
                    "OpenAI call failed with %s; retrying without it",
                    param,
                    extra={"model": model},
                    exc_info=exc,
                )
                kwargs.pop(param, None)
                continue
            raise
    return _message_content_to_text(response.choices[0].message)


def _groq_chat_completion(
    *,
    model: str,
    messages: list[dict[str, str]],
    use_json_mode: bool,
    reasoning_effort: str | None = None,
):
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }
    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    # GPT-OSS is a reasoning model on Groq and accepts an effort hint; Llama does
    # not. The pinned Groq SDK doesn't type `reasoning_effort`, so forward it via
    # extra_body (it reaches the API as a request-body field) instead of as a
    # top-level kwarg, which the SDK rejects with a TypeError.
    if reasoning_effort:
        kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
    try:
        return get_groq_client().chat.completions.create(**kwargs)
    except Exception as exc:
        if _is_provider_auth_error(exc):
            _raise_provider_auth_error("Groq", "GROQ_API_KEY", exc)
        if not use_json_mode or not _is_response_format_error(exc):
            raise
        logger.info(
            "Groq JSON response_format unsupported; retrying without response_format",
            extra={"model": model, "error": str(exc)},
        )
        retry_kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0}
        if reasoning_effort:
            retry_kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
        try:
            return get_groq_client().chat.completions.create(**retry_kwargs)
        except Exception as retry_exc:
            if _is_provider_auth_error(retry_exc):
                _raise_provider_auth_error("Groq", "GROQ_API_KEY", retry_exc)
            raise


# ---- client construction (lazy) ----------------------------------------------

# Clients are built once and reused (``lru_cache``) so repeated per-dimension
# calls reuse one HTTP connection pool (keep-alive) instead of a fresh TLS
# handshake each time. Construction is lazy and a missing key raises (not
# cached), so the app/worker still boot when a given provider's key is unset —
# an unset key fails only that provider's path, not the whole process.

@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Please contact administrators."
        )
    return OpenAI(api_key=api_key)


@lru_cache(maxsize=1)
def get_deepseek_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing DEEPSEEK_API_KEY. Please contact administrators."
        )
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


@lru_cache(maxsize=1)
def get_groq_client() -> "Groq":
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GROQ_API_KEY. Please contact administrators."
        )
    return Groq(api_key=api_key)


@lru_cache(maxsize=1)
def get_claude_client() -> "Anthropic":
    if Anthropic is None:
        raise RuntimeError(
            "The anthropic SDK is not installed. Add 'anthropic' to requirements.txt."
        )
    api_key = os.environ.get("CLAUDE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing CLAUDE_API_KEY. Please contact administrators."
        )
    return Anthropic(api_key=api_key)


# ---- Claude (Anthropic) chat helpers -----------------------------------------

def _split_system_for_anthropic(
    messages: list[dict[str, str]]
) -> tuple[str, list[dict[str, str]]]:
    """Anthropic takes the system prompt as a separate argument, so pull any
    'system'-role messages out of the OpenAI-style messages list."""
    system_parts = [
        str(m.get("content") or "") for m in messages if m.get("role") == "system"
    ]
    convo = [
        {"role": m.get("role", "user"), "content": str(m.get("content") or "")}
        for m in messages
        if m.get("role") != "system"
    ]
    return "\n\n".join(p for p in system_parts if p).strip(), convo


def _claude_response_text(response: Any) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts).strip()


def _claude_chat(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int | None = None,
) -> str:
    """Call Claude (Anthropic Messages API) and return the concatenated text.

    Mirrors the synchronous client pattern used for Groq/DeepSeek. Reasoning
    effort is OpenAI-family-specific and intentionally not forwarded here."""
    client = get_claude_client()
    system, convo = _split_system_for_anthropic(messages)
    # NB: no `temperature` — Opus 4.8 (the default) rejects it as deprecated, and
    # it is optional for every Claude model, so omitting it is universally safe.
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens or _claude_max_tokens(),
        "messages": convo or [{"role": "user", "content": ""}],
    }
    if system:
        kwargs["system"] = system
    try:
        response = client.messages.create(**kwargs)
    except Exception as exc:
        if _is_provider_auth_error(exc):
            _raise_provider_auth_error("Claude", "CLAUDE_API_KEY", exc)
        raise
    return _claude_response_text(response)
