"""Rate limiting — Dev Plan v2 §6 (llm/ratelimit.py).

Per-minute limits (requests and tokens) are enforced by token buckets, so the client
paces itself instead of collecting 429 errors. Per-day limits and the budget are
checked against the usage ledger before every paid call.
"""
from __future__ import annotations

from aiolimiter import AsyncLimiter


class DailyLimitReached(RuntimeError):
    """The provider's per-day request or token limit would be exceeded. Resume tomorrow (UTC)."""


class BudgetExceeded(RuntimeError):
    """The configured USD budget would be exceeded. Raise LLM_BUDGET_USD deliberately to continue."""


class MinuteLimiter:
    def __init__(self, rpm: int, tpm: int):
        self.req = AsyncLimiter(max_rate=rpm, time_period=60)
        self.tok = AsyncLimiter(max_rate=tpm, time_period=60)
        self.tpm = tpm

    async def acquire(self, est_tokens: int) -> None:
        await self.req.acquire()
        await self.tok.acquire(min(max(est_tokens, 1), self.tpm))


def estimate_tokens(messages: list[dict], max_tokens: int, chars_per_token: float = 2.5) -> int:
    """Conservative pre-call estimate for the token bucket (Sinhala tokenises densely).

    Calibrated in notebook 03 §6 from measured usage; only used for pacing, never for results.
    """
    chars = sum(len(m.get("content") or "") for m in messages)
    return int(chars / chars_per_token) + min(max_tokens, 800)