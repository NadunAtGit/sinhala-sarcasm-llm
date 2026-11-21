"""Async LLM client — Dev Plan v2 §6 (llm/client.py).

- OpenAI-compatible (Groq): provider, model and limits come from .env, never from code.
- Every call is cached by request hash; a repeated request costs nothing.
- Retries with exponential back-off on 429 / timeouts / 5xx (tenacity).
- Paces itself to the per-minute limits; refuses calls that would break the
  per-day limits or the USD budget (checked against the usage ledger).
- Saves the model string *returned by the provider* with every response (provider drift control).
"""
from __future__ import annotations

import asyncio
import datetime as dt
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import openai
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from .cache import Store, request_key, utc_day
from .ratelimit import BudgetExceeded, DailyLimitReached, MinuteLimiter, estimate_tokens

RETRYABLE = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError,
             openai.InternalServerError)


@dataclass
class LLMSettings:
    api_key: str
    base_url: str
    model: str
    rpm: int = 30
    tpm: int = 8_000
    rpd: int = 1_000
    tpd: int = 200_000
    budget_usd: float = 5.0
    price_in: float = 0.15      # USD per 1M input tokens
    price_out: float = 0.60     # USD per 1M output tokens
    max_concurrency: int = 4
    timeout_s: float = 120.0

    @classmethod
    def from_env(cls, env_path: str | Path | None = None) -> "LLMSettings":
        if env_path:
            load_dotenv(env_path, override=True)
        g = os.environ.get
        key = g("LLM_API_KEY", "")
        if not key or key.startswith("gsk_your") or key.startswith("gsk_paste"):
            raise RuntimeError("LLM_API_KEY is not set — put your Groq key in .env")
        return cls(
            api_key=key, base_url=g("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
            model=g("LLM_MODEL", "openai/gpt-oss-120b"),
            rpm=int(g("LLM_RPM", 30)), tpm=int(g("LLM_TPM", 8000)),
            rpd=int(g("LLM_RPD", 1000)), tpd=int(g("LLM_TPD", 200000)),
            budget_usd=float(g("LLM_BUDGET_USD", 5)),
            price_in=float(g("LLM_PRICE_IN", 0.15)), price_out=float(g("LLM_PRICE_OUT", 0.60)),
            max_concurrency=int(g("LLM_MAX_CONCURRENCY", 4)), timeout_s=float(g("LLM_TIMEOUT_S", 120)),
        )

    def public(self) -> dict:
        d = asdict(self)
        d["api_key"] = self.api_key[:6] + "…" + self.api_key[-4:]
        return d


class LLMClient:
    def __init__(self, settings: LLMSettings, cache_path: str | Path, http_client=None):
        self.s = settings
        self.store = Store(cache_path)
        self.api = AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url,
                               timeout=settings.timeout_s, max_retries=0, http_client=http_client)
        self.limiter = MinuteLimiter(settings.rpm, settings.tpm)
        self.sem = asyncio.Semaphore(settings.max_concurrency)
        self._lock = asyncio.Lock()   # makes the budget/daily check + reservation atomic
        self._reserved = 0.0

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return tokens_in / 1e6 * self.s.price_in + tokens_out / 1e6 * self.s.price_out

    async def _guard(self, est_in: int, max_tokens: int) -> float:
        """Check budget and daily limits before a paid call; reserve the worst-case cost."""
        worst = self.cost(est_in, max_tokens)
        async with self._lock:
            spent = self.store.totals()["cost_usd"]
            if spent + self._reserved + worst > self.s.budget_usd:
                raise BudgetExceeded(f"spent ${spent:.4f} + reserved ${self._reserved:.4f} "
                                     f"+ this call ≤ ${worst:.4f} > budget ${self.s.budget_usd:.2f}")
            today = self.store.totals(utc_day())
            if today["requests"] >= self.s.rpd:
                raise DailyLimitReached(f"{today['requests']} requests today ≥ RPD {self.s.rpd}")
            if today["tokens_in"] + today["tokens_out"] + est_in + max_tokens > self.s.tpd:
                raise DailyLimitReached(f"{today['tokens_in'] + today['tokens_out']:,} tokens today; "
                                        f"next call could exceed TPD {self.s.tpd:,}")
            self._reserved += worst
        return worst

    async def chat(self, messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 1024,
                   reasoning_effort: str | None = "low", json_mode: bool = True,
                   use_cache: bool = True, tag: str = "") -> dict:
        params = {"temperature": temperature, "max_tokens": max_tokens,
                  "reasoning_effort": reasoning_effort, "json_mode": json_mode}
        key = request_key(self.s.model, messages, params)
        if use_cache and (hit := self.store.get(key)):
            return {**hit, "cache_hit": True}

        est = estimate_tokens(messages, max_tokens)
        reserved = await self._guard(est, max_tokens)
        try:
            async with self.sem:
                await self.limiter.acquire(est)
                kwargs = dict(model=self.s.model, messages=messages, temperature=temperature,
                              max_completion_tokens=max_tokens)
                if reasoning_effort:
                    kwargs["reasoning_effort"] = reasoning_effort
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                t0 = time.perf_counter()
                async for attempt in AsyncRetrying(retry=retry_if_exception_type(RETRYABLE),
                                                   wait=wait_random_exponential(multiplier=2, max=60),
                                                   stop=stop_after_attempt(6), reraise=True):
                    with attempt:
                        resp = await self.api.chat.completions.create(**kwargs)
                latency = time.perf_counter() - t0
        finally:
            async with self._lock:
                self._reserved -= reserved

        msg = resp.choices[0].message
        extra = getattr(msg, "model_extra", None) or {}
        reasoning = getattr(msg, "reasoning", None) or extra.get("reasoning")
        tin = resp.usage.prompt_tokens if resp.usage else 0
        tout = resp.usage.completion_tokens if resp.usage else 0
        record = {
            "key": key, "content": msg.content or "", "reasoning": reasoning,
            "model": resp.model, "finish_reason": resp.choices[0].finish_reason,
            "tokens_in": tin, "tokens_out": tout, "cost_usd": self.cost(tin, tout),
            "latency_s": round(latency, 3), "params": params,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        self.store.log_usage(resp.model, tag, tin, tout, record["cost_usd"], key)
        if use_cache:
            self.store.put(key, record)
        return {**record, "cache_hit": False}

    async def map(self, batch: list[list[dict]], **kw) -> list[dict | Exception]:
        """Run many requests concurrently (bounded); failures are returned, not raised."""
        return await asyncio.gather(*(self.chat(m, **kw) for m in batch), return_exceptions=True)

    def usage(self) -> dict:
        return {"all_time": self.store.totals(), "today_utc": self.store.totals(utc_day()),
                "cached_responses": self.store.n_cached()}