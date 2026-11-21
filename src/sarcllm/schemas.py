"""Output schemas and robust parsing — Dev Plan v2 §5.3, §13 (test_parse).

Parse failures are never silently fixed: a row that cannot be parsed is returned with
ok=False and counts as wrong in the scorer (and is reported separately).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

Sentiment = Literal["negative", "neutral", "positive"]

_SENT = {"negative": "negative", "neg": "negative", "neutral": "neutral", "neu": "neutral",
         "positive": "positive", "pos": "positive"}
_TRUE = {"true", "yes", "1", "sarcastic", "y"}
_FALSE = {"false", "no", "0", "non-sarcastic", "not sarcastic", "non_sarcastic", "n"}


def norm_sentiment(v):
    if isinstance(v, str) and v.strip().lower() in _SENT:
        return _SENT[v.strip().lower()]
    return v   # leave invalid values for pydantic to reject


def norm_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
    return v


class FinalOutput(BaseModel):
    literal_sentiment: Sentiment
    is_sarcastic: bool
    intended_sentiment: Sentiment
    rationale: str = ""
    evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("literal_sentiment", "intended_sentiment", mode="before")
    @classmethod
    def _sent(cls, v):
        return norm_sentiment(v)

    @field_validator("is_sarcastic", mode="before")
    @classmethod
    def _bool(cls, v):
        return norm_bool(v)

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def _ids(cls, v):
        if v is None:
            return []
        return [str(x) for x in v] if isinstance(v, list) else [str(v)]


class AgentOutput(BaseModel):
    dimension: str
    judgement: str
    confidence: float = Field(ge=0, le=1)
    evidence: str | list[str] = ""


@dataclass
class ParseResult:
    ok: bool
    obj: BaseModel | None
    error: str = ""


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> dict:
    """First JSON object in `text`: handles code fences and leading/trailing prose."""
    if not text or not text.strip():
        raise ValueError("empty response")
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unbalanced JSON object (truncated output?)")


def _parse(text: str, model: type[BaseModel]) -> ParseResult:
    try:
        return ParseResult(True, model.model_validate(extract_json(text)))
    except (ValueError, json.JSONDecodeError, ValidationError) as e:
        return ParseResult(False, None, f"{type(e).__name__}: {str(e)[:200]}")


def parse_final(text: str) -> ParseResult:
    return _parse(text, FinalOutput)


def parse_agent(text: str) -> ParseResult:
    return _parse(text, AgentOutput)