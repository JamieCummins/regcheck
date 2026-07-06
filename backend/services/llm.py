"""LLM provider layer for RegCheck comparisons.

Everything that talks to a model provider lives here: client construction
(OpenAI / DeepSeek / Claude), per-provider model resolution, the chat
helpers used by the comparison flows, provider-error classification, and the
response-parsing utilities. Clients are built lazily so the app/worker boot even
when a given provider's key is unset — an unset key fails only that provider's
path, not the whole process.

This module is intentionally free of comparison/business logic; ``comparisons``
imports these helpers and orchestrates them.
"""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any

from openai import OpenAI

try:  # Optional provider SDK; only required when client_choice == "claude".
    from anthropic import Anthropic
except ImportError:  # pragma: no cover - dependency optional at runtime
    Anthropic = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_MODEL = "gpt-5.5"
DEFAULT_DEEPSEEK_MODEL = "deepseek-reasoner"
DEFAULT_CLAUDE_MODEL = "claude-opus-4-8"
# Open-weight Qwen, served via Groq's OpenAI-compatible endpoint (uses GROQ_API_KEY).
DEFAULT_QWEN_MODEL = "qwen/qwen3.6-27b"
# Uni Bern GPUStack — OpenAI-compatible, but reachable only from inside the Bern
# network. So the gpustack provider only works from a machine on that network
# (e.g. the CLI run locally, or an in-network worker), not the Heroku worker.
DEFAULT_GPUSTACK_BASE_URL = "https://gpustack.unibe.ch/v1"
DEFAULT_GPUSTACK_MODEL = "gpt-oss-120b"

# Client choices that run through the OpenAI SDK against the *hosted* OpenAI API.
# (Qwen also uses the OpenAI SDK but against Groq's base URL — see get_groq_openai_client.)
_OPENAI_CLIENTS = {"openai"}

# Providers usable from the HOSTED app/API (the Heroku worker can reach them).
# gpustack is intentionally excluded: it is reachable only from inside the Uni Bern
# network, so it is offered on the local CLI only. The CLI accepts ALL_CLIENTS.
HOSTED_CLIENTS = frozenset({"openai", "deepseek", "qwen", "claude"})
ALL_CLIENTS = HOSTED_CLIENTS | {"gpustack"}


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


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(str(raw).strip())
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


# ---- per-provider model resolution -------------------------------------------

def _openai_model() -> str:
    return _env_str("OPENAI_COMPARISON_MODEL", _env_str("OPENAI_MODEL", DEFAULT_OPENAI_MODEL))


def _openai_experiment_model() -> str:
    return _env_str("OPENAI_EXPERIMENT_MODEL", _openai_model())


def _openai_family_model(client_choice: str, *, experiment: bool = False) -> str:
    return _openai_experiment_model() if experiment else _openai_model()


def _deepseek_model() -> str:
    return _env_str("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)


def _qwen_model() -> str:
    return _env_str("QWEN_MODEL", DEFAULT_QWEN_MODEL)


def _gpustack_model() -> str:
    return _env_str("GPUSTACK_MODEL", DEFAULT_GPUSTACK_MODEL)


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


def _qwen_chat(
    messages: list[dict[str, str]], *, model: str | None = None, use_json_mode: bool = False
) -> str:
    """Qwen 3.6 27B (open-weight) via Groq's OpenAI-compatible endpoint.

    ``temperature=0.6``; reasoning effort from ``QWEN_REASONING_EFFORT`` (default
    ``"none"``). When ``use_json_mode`` (the comparison call), request JSON-object
    structured output (``response_format={"type":"json_object"}``) — verified
    supported + reliable here, and it stops the occasional prose-wrapped/truncated
    reply that the plain-text path let through. The experiment-isolation call leaves
    it off (it needs free text, not a JSON object). Strict ``json_schema`` is NOT
    used — this Groq model rejects it (400).

    IMPORTANT: we do NOT set Groq's ``reasoning_format`` — for this model
    "hidden"/"parsed" route the *answer* into the reasoning channel and return empty
    ``content`` (verified live). With no reasoning_format the answer is in
    ``content``; ``reasoning_effort="none"`` keeps it tight + complete, and
    ``_strip_deepseek_reasoning`` strips any stray reasoning.
    """
    client = get_groq_openai_client()
    kwargs: dict[str, Any] = {
        "model": model or _qwen_model(),
        "messages": messages,
        "temperature": 0.6,
        "reasoning_effort": _env_str("QWEN_REASONING_EFFORT", "none"),
    }
    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    while True:
        try:
            response = client.chat.completions.create(**kwargs)
            break
        except Exception as exc:
            if _is_provider_auth_error(exc):
                _raise_provider_auth_error("Groq", "GROQ_API_KEY", exc)
            # Drop an optional knob the endpoint rejects and retry, rather than
            # failing the dimension outright (Groq model support varies).
            lowered = str(exc).lower()
            dropped = None
            if ("response_format" in lowered or "json" in lowered) and "response_format" in kwargs:
                kwargs.pop("response_format"); dropped = "response_format"
            elif "reasoning_effort" in lowered and "reasoning_effort" in kwargs:
                kwargs.pop("reasoning_effort"); dropped = "reasoning_effort"
            elif "temperature" in lowered and "temperature" in kwargs:
                kwargs.pop("temperature"); dropped = "temperature"
            if dropped is None:
                raise
            logger.info("Qwen call rejected %s; retrying without it", dropped, exc_info=exc)
    raw = _message_content_to_text(response.choices[0].message)
    return _strip_deepseek_reasoning(raw)


def _gpustack_chat(
    messages: list[dict[str, str]], *, model: str | None = None, use_json_mode: bool = False
) -> str:
    """gpt-oss-120b (open-weight) via Uni Bern's GPUStack OpenAI-compatible endpoint.

    GPUStack is firewalled to the Bern network, so this only succeeds from a machine
    inside that network. Sampling mirrors GPUStack's reference snippet
    (``temperature=1``, ``top_p=1``) — appropriate defaults for a reasoning model
    like gpt-oss — and every knob is env-overridable (``GPUSTACK_TEMPERATURE``,
    ``GPUSTACK_TOP_P``, ``GPUSTACK_MAX_TOKENS``, ``GPUSTACK_REASONING_EFFORT``).

    IMPORTANT: we do NOT set ``response_format={"type":"json_object"}``. gpt-oss-120b's
    JSON-object guided decoding is broken on this GPUStack deployment — it leaks
    harmony control tokens (e.g. ``<|constrain|>``) and returns invalid JSON (verified
    live). Plain generation, by contrast, yields clean parseable JSON. So when
    ``use_json_mode`` (the comparison call) we append a short JSON-only instruction and
    rely on the comparison prompt's explicit "single JSON object" requirement plus
    ``_extract_json_payload``; the experiment-isolation call (free text) leaves it off.
    Any optional knob the endpoint rejects is dropped and the call retried, and stray
    ``<think>`` reasoning is stripped from the reply (mirrors the Qwen path).
    """
    client = get_gpustack_client()
    chat_messages = list(messages)
    if use_json_mode:
        chat_messages.append(
            {
                "role": "system",
                "content": (
                    "Respond with ONLY a single valid JSON object — no prose, no "
                    "markdown fences, no extra text before or after."
                ),
            }
        )
    kwargs: dict[str, Any] = {
        "model": model or _gpustack_model(),
        "messages": chat_messages,
        "temperature": _env_float("GPUSTACK_TEMPERATURE", 1.0),
        "top_p": _env_float("GPUSTACK_TOP_P", 1.0),
    }
    max_tokens = _env_int("GPUSTACK_MAX_TOKENS", 40000)
    if max_tokens > 0:
        kwargs["max_tokens"] = max_tokens
    effort = (os.environ.get("GPUSTACK_REASONING_EFFORT") or "").strip()
    if effort:
        kwargs["reasoning_effort"] = effort
    while True:
        try:
            response = client.chat.completions.create(**kwargs)
            break
        except Exception as exc:
            if _is_provider_auth_error(exc):
                _raise_provider_auth_error("GPUStack", "GPUSTACK_API_KEY", exc)
            # Endpoint/model support varies; drop a rejected optional knob and retry
            # rather than failing the dimension outright.
            lowered = str(exc).lower()
            dropped = None
            for knob in ("reasoning_effort", "max_tokens", "top_p", "temperature"):
                if knob in lowered and knob in kwargs:
                    kwargs.pop(knob)
                    dropped = knob
                    break
            if dropped is None:
                raise
            logger.info("GPUStack call rejected %s; retrying without it", dropped, exc_info=exc)
    raw = _message_content_to_text(response.choices[0].message)
    return _strip_deepseek_reasoning(raw)


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
def get_groq_openai_client() -> OpenAI:
    """Groq's OpenAI-compatible endpoint, via the OpenAI SDK (used for Qwen).

    ``reasoning_effort``/``temperature`` are native OpenAI args here; Groq-specific
    knobs (e.g. ``reasoning_format``) are forwarded via ``extra_body``."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GROQ_API_KEY. Please contact administrators."
        )
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


@lru_cache(maxsize=1)
def get_gpustack_client() -> OpenAI:
    """Uni Bern GPUStack via the OpenAI SDK. Reachable only inside the Bern network.

    Base URL defaults to the campus endpoint but is overridable via
    ``GPUSTACK_BASE_URL`` (the SDK appends ``/chat/completions`` etc.)."""
    api_key = os.environ.get("GPUSTACK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GPUSTACK_API_KEY. Set it in your environment (.env) to use GPUStack models."
        )
    base_url = _env_str("GPUSTACK_BASE_URL", DEFAULT_GPUSTACK_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url)


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

    Mirrors the synchronous client pattern used for DeepSeek. Reasoning
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


def _claude_structured(
    messages: list[dict[str, str]],
    *,
    model: str,
    tool: dict[str, Any],
    max_tokens: int | None = None,
) -> str:
    """Call Claude with a FORCED tool call and return the tool input as a JSON string.

    Claude has no `response_format` schema mode like OpenAI's `.parse()`; left to plain
    text it occasionally emits malformed JSON (e.g. an unescaped quote inside a value),
    which `json.loads` then rejects. Forcing a single tool (``tool_choice`` = that tool)
    makes Anthropic emit the arguments as a structured `tool_use` block whose ``.input``
    is an already-parsed dict — schema-constrained at decode time, so the malformed-JSON
    failure mode disappears. We re-serialise with ``json.dumps`` so callers keep their
    existing "string in → json.loads" contract."""
    client = get_claude_client()
    system, convo = _split_system_for_anthropic(messages)
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens or _claude_max_tokens(),
        "messages": convo or [{"role": "user", "content": ""}],
        "tools": [tool],
        "tool_choice": {"type": "tool", "name": tool["name"]},
    }
    if system:
        kwargs["system"] = system
    try:
        response = client.messages.create(**kwargs)
    except Exception as exc:
        if _is_provider_auth_error(exc):
            _raise_provider_auth_error("Claude", "CLAUDE_API_KEY", exc)
        raise
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == tool["name"]:
            try:
                return json.dumps(block.input, ensure_ascii=False)
            except (TypeError, ValueError):
                return json.dumps(dict(block.input or {}), ensure_ascii=False)
    # Forced tool_choice should always yield a tool_use block; fall back to text just in case.
    return _claude_response_text(response)
