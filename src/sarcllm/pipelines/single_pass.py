"""Single-pass pipeline: L1, L2, L3 and the HY3 resolver — Dev Plan v2 §6.

One call per item; if the reply is not valid JSON for the schema, ONE repair call is made
(the model sees its own reply and a short correction request). If that also fails, the
item is recorded with parse_ok = False and counts as wrong in the scorer.
"""
from __future__ import annotations

from ..prompts import Prompts
from ..schemas import parse_final


def _prediction(item_id: str, condition: str, parsed, calls: list[dict], allowed_ids: set[str],
                retrieved_ids: list[str] | None = None, error: str = "") -> dict:
    last = calls[-1]
    ok = parsed is not None
    ev = list(parsed.evidence_ids) if ok else []
    return {
        "id": item_id, "condition": condition,
        "is_sarcastic_pred": int(parsed.is_sarcastic) if ok else None,
        "sentiment_pred": parsed.intended_sentiment if ok else None,
        "literal_sentiment_pred": parsed.literal_sentiment if ok else None,
        "rationale": parsed.rationale if ok else "",
        "evidence_ids": ev, "evidence_valid": all(e in allowed_ids for e in ev),
        "retrieved_ids": retrieved_ids or [],
        "parse_ok": ok, "parse_error": error,
        "n_calls": len(calls), "n_paid_calls": sum(not c["cache_hit"] for c in calls),
        "tokens_in": sum(c["tokens_in"] for c in calls), "tokens_out": sum(c["tokens_out"] for c in calls),
        "cost_usd": sum(c["cost_usd"] for c in calls if not c["cache_hit"]),
        "latency_s": sum(c["latency_s"] for c in calls),
        "cache_hit": all(c["cache_hit"] for c in calls),
        "model": last["model"], "finish_reason": last["finish_reason"],
        "response_keys": [c["key"] for c in calls],
    }


async def run_item(client, prompts: Prompts, item: dict, *, condition: str, gen: dict,
                   examples: list[dict] = (), note: str | None = None,
                   retrieved_ids: list[str] | None = None) -> dict:
    messages = prompts.single_pass(item, examples, note)
    allowed = {ex["id"] for ex in examples}
    calls = [await client.chat(messages, tag=condition, **gen)]
    res = parse_final(calls[-1]["content"])
    if not res.ok:
        repair = messages + [{"role": "assistant", "content": calls[-1]["content"] or "(empty)"},
                             {"role": "user", "content": prompts.repair(res.error)}]
        calls.append(await client.chat(repair, tag=f"{condition}-repair", **gen))
        res2 = parse_final(calls[-1]["content"])
        if res2.ok:
            res = res2
        else:
            return _prediction(item["id"], condition, None, calls, allowed, retrieved_ids,
                               error=f"{res.error} | after repair: {res2.error}")
    return _prediction(item["id"], condition, res.obj, calls, allowed, retrieved_ids)