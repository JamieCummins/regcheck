"""Live per-run LLM cost tracking + reasoning (chain-of-thought) capture.

A ``CostTracker`` is created per report run (orchestrator) and stored in a
ContextVar: ``asyncio.to_thread`` copies the context into worker threads, and
the copied variable still references the SAME tracker object, so provider
helpers called anywhere inside the run can record token usage without any
plumbing through call signatures. The tracker itself is lock-guarded, so
concurrent dimension threads within a run stay consistent, and concurrent runs
(separate worker tasks) each see their own tracker.

USD figures are ESTIMATES from a per-1M-token pricing table (prefix-matched on
the model name). Prices drift — override or extend at deploy time with
``COST_PRICING_JSON`` (e.g. '{"gpt-5.5": [1.25, 10.0]}' for input/output, or
'{"text-embedding-3-large": 0.13}' for embeddings). Unknown models still have
their tokens counted, but the snapshot is flagged ``estimate_complete: false``.

Reasoning capture: providers that expose chain-of-thought (DeepSeek's
``reasoning_content``, or inline ``<think>…</think>`` from open-weight models)
stash it thread-locally; the judgement parser pops it onto the item's
``chain_of_thought`` field. Same-thread stash/pop, so nothing leaks across
dimensions.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from contextvars import ContextVar
from typing import Any

logger = logging.getLogger(__name__)

# Per-1M-token USD prices. LLMs: (input, output). Embeddings: single float.
# APPROXIMATE — verify against provider pricing pages; override via COST_PRICING_JSON.
_DEFAULT_PRICING: dict[str, Any] = {
    "gpt-5.5": (1.25, 10.0),
    "gpt-5": (1.25, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.0),
    "deepseek-reasoner": (0.55, 2.19),
    "deepseek-chat": (0.27, 1.10),
    "claude-opus": (15.0, 75.0),
    "claude-sonnet": (3.0, 15.0),
    "text-embedding-3-large": 0.13,
    "text-embedding-3-small": 0.02,
}


def _pricing() -> dict[str, Any]:
    table = dict(_DEFAULT_PRICING)
    raw = (os.environ.get("COST_PRICING_JSON") or "").strip()
    if raw:
        try:
            override = json.loads(raw)
            if isinstance(override, dict):
                table.update({str(k).lower(): v for k, v in override.items()})
        except (ValueError, TypeError):  # pragma: no cover - defensive
            logger.warning("Ignoring malformed COST_PRICING_JSON")
    return table


def _match_price(model: str) -> Any | None:
    """Longest-prefix match so e.g. 'gpt-5.5-2026-05' hits the 'gpt-5.5' row."""
    name = (model or "").strip().lower()
    best_key, best = "", None
    for key, value in _pricing().items():
        if name.startswith(key) and len(key) > len(best_key):
            best_key, best = key, value
    return best


class CostTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0
        self.embedding_tokens = 0
        self.llm_usd = 0.0
        self.embedding_usd = 0.0
        self.llm_calls = 0
        self.models: set[str] = set()
        self.unpriced_models: set[str] = set()

    def record_llm(self, model: str, input_tokens: int, output_tokens: int) -> None:
        price = _match_price(model)
        with self._lock:
            self.llm_calls += 1
            self.input_tokens += max(0, int(input_tokens or 0))
            self.output_tokens += max(0, int(output_tokens or 0))
            if model:
                self.models.add(model)
            if isinstance(price, (list, tuple)) and len(price) == 2:
                self.llm_usd += (
                    max(0, int(input_tokens or 0)) * float(price[0])
                    + max(0, int(output_tokens or 0)) * float(price[1])
                ) / 1_000_000
            elif model:
                self.unpriced_models.add(model)

    def record_embedding(self, model: str, tokens: int) -> None:
        price = _match_price(model)
        with self._lock:
            self.embedding_tokens += max(0, int(tokens or 0))
            if model:
                self.models.add(model)
            if isinstance(price, (int, float)):
                self.embedding_usd += max(0, int(tokens or 0)) * float(price) / 1_000_000
            elif model:
                self.unpriced_models.add(model)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "embedding_tokens": self.embedding_tokens,
                "llm_calls": self.llm_calls,
                "llm_usd": round(self.llm_usd, 4),
                "embedding_usd": round(self.embedding_usd, 4),
                "total_usd": round(self.llm_usd + self.embedding_usd, 4),
                "estimate_complete": not self.unpriced_models,
                "models": sorted(self.models),
            }


_ACTIVE: ContextVar[CostTracker | None] = ContextVar("regcheck_cost_tracker", default=None)


def start_run() -> CostTracker:
    """Create and activate a fresh tracker for the current run's context."""
    tracker = CostTracker()
    _ACTIVE.set(tracker)
    return tracker


def current() -> CostTracker | None:
    return _ACTIVE.get()


def record_llm_usage(model: str, usage: Any) -> None:
    """Record a completion's token usage from an OpenAI-style (prompt_tokens /
    completion_tokens) or Anthropic-style (input_tokens / output_tokens) usage
    object. No-op when no tracker is active or usage is absent."""
    tracker = current()
    if tracker is None or usage is None:
        return
    input_tokens = getattr(usage, "prompt_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(usage, "input_tokens", 0)
    output_tokens = getattr(usage, "completion_tokens", None)
    if output_tokens is None:
        output_tokens = getattr(usage, "output_tokens", 0)
    tracker.record_llm(model, input_tokens or 0, output_tokens or 0)


def record_embedding_usage(model: str, usage: Any, *, tokens: int | None = None) -> None:
    tracker = current()
    if tracker is None:
        return
    if tokens is None:
        tokens = getattr(usage, "total_tokens", 0) if usage is not None else 0
    tracker.record_embedding(model, tokens or 0)


# ── reasoning (chain-of-thought) stash — same-thread set/pop ───────────────────

_REASONING = threading.local()


def stash_reasoning(text: str | None) -> None:
    """Store reasoning text emitted alongside a completion (providers that expose
    it). Kept thread-local: the judgement parser pops it in the same thread."""
    if text and str(text).strip():
        _REASONING.value = str(text).strip()


def pop_reasoning() -> str:
    value = getattr(_REASONING, "value", "") or ""
    _REASONING.value = ""
    return value
