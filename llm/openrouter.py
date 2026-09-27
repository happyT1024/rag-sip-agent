"""OpenRouter chat-completion client with cost tracking and a hard budget cap.

The key is read from the ``OPENROUTER_API_KEY`` environment variable (never
logged, never echoed). Every call returns a :class:`Reply` carrying the
spend, and the client keeps a running total; when the total crosses
``budget_usd`` the client raises :class:`BudgetExceeded` instead of silently
burning money.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"
PRICING_CACHE = Path("data/cache/openrouter_models.json")


class BudgetExceeded(RuntimeError):
    """Raised before a call that would push the running spend over budget."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.cost_usd + other.cost_usd,
        )


@dataclass
class Reply:
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (see .env.example)")
    return key


def load_pricing(path: Path = PRICING_CACHE, refresh: bool = False) -> dict[str, dict]:
    """model id -> {'prompt': $/token, 'completion': $/token}; cached on disk."""
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    req = urllib.request.Request(MODELS_URL, headers={"User-Agent": "rag-sip-agent"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = json.load(resp)["data"]
    pricing = {m["id"]: m.get("pricing", {}) for m in raw}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pricing))
    return pricing


def estimate_cost(model: str, usage: Usage, pricing: dict[str, dict] | None = None) -> float:
    """Fallback cost calc from per-token prices when the API omits `usage.cost`."""
    pricing = pricing if pricing is not None else load_pricing()
    p = pricing.get(model, {})
    pin = float(p.get("prompt", 0.0) or 0.0)
    pout = float(p.get("completion", 0.0) or 0.0)
    return usage.prompt_tokens * pin + usage.completion_tokens * pout


class OpenRouterClient:
    """Thin blocking client; ``temperature=0`` and short answers keep cost low."""

    def __init__(
        self,
        api_key: str | None = None,
        budget_usd: float = 1.50,
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key or _api_key()
        self.budget_usd = budget_usd
        self.max_retries = max_retries
        self.spend_usd = 0.0
        self.usage = Usage()

    def chat(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> Reply:
        if self.spend_usd >= self.budget_usd:
            raise BudgetExceeded(
                f"run budget exhausted: ${self.spend_usd:.4f} >= ${self.budget_usd:.2f}"
            )
        payload: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "usage": {"include": True},
        }
        if tools:
            payload["tools"] = tools

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._request(model, payload)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
                last_err = err
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"OpenRouter request failed after {self.max_retries} retries: {last_err}")

    def _request(self, model: str, payload: dict) -> Reply:
        req = urllib.request.Request(
            API_URL,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "rag-sip-agent",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as err:
            body = err.read().decode(errors="replace")
            # Budget/credit errors must not be retried blindly with the same payload.
            raise RuntimeError(f"OpenRouter HTTP {err.code}: {body[:300]}") from err

        choice = data["choices"][0]
        message = choice.get("message", {})
        raw = data.get("usage", {}) or {}
        usage = Usage(
            prompt_tokens=int(raw.get("prompt_tokens", 0) or 0),
            completion_tokens=int(raw.get("completion_tokens", 0) or 0),
        )
        cost = raw.get("cost")
        usage.cost_usd = float(cost) if cost is not None else estimate_cost(model, usage)
        self.usage = self.usage + usage
        self.spend_usd += usage.cost_usd

        content = message.get("content") or ""
        tool_calls = message.get("tool_calls") or []
        return Reply(content=content, tool_calls=tool_calls, usage=usage, model=model)
