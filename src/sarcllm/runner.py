"""Batch runner — Dev Plan v2 §6 (runner.py), §7, §12.

- One folder per run: runs/<run_id>/{predictions.jsonl, errors.jsonl, meta.json}.
- Resume: items already in predictions.jsonl are skipped; only missing/failed items are run.
- Each prediction is written as soon as it completes (an interruption loses nothing).
- Test guard: refuses split="test" unless configs/base.yaml has frozen: true (Gate G1).
- Budget / daily-limit stops end the run cleanly; meta.json records every session.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Awaitable, Callable

import pandas as pd

from .llm.ratelimit import BudgetExceeded, DailyLimitReached


class FrozenGuardError(RuntimeError):
    """Test-set runs are blocked until the design is frozen at Gate G1."""


def git_commit(root: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return (out.stdout.strip() or "unknown") + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def load_predictions(run_dir: Path) -> pd.DataFrame:
    p = Path(run_dir) / "predictions.jsonl"
    if not p.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return pd.DataFrame(rows).drop_duplicates("id", keep="last") if rows else pd.DataFrame()


async def run(*, cond: dict, base: dict, rows: pd.DataFrame, split: str, set_name: str,
              client, item_fn: Callable[[object], Awaitable[dict]], root: str | Path,
              prompt_version: str, run_id: str | None = None, extra_meta: dict | None = None) -> pd.DataFrame:
    root = Path(root)
    if split == "test" and not base.get("frozen", False):
        raise FrozenGuardError("split='test' is blocked: set frozen: true in configs/base.yaml at Gate G1")
    if not set(rows["split"]) <= {split}:
        raise ValueError(f"rows contain splits {set(rows['split'])}, expected only {split!r}")

    run_id = run_id or f"{cond['id'].lower()}_{split}_{set_name}"
    run_dir = root / base["paths"]["runs"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_path, err_path, meta_path = run_dir / "predictions.jsonl", run_dir / "errors.jsonl", run_dir / "meta.json"

    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {
        "run_id": run_id, "condition": cond, "split": split, "set_name": set_name,
        "n_items": len(rows), "generation": base["generation"], "input": base.get("input"),
        "prompt_version": prompt_version, "sessions": [], **(extra_meta or {})}
    if meta["prompt_version"] != prompt_version:
        raise RuntimeError(f"run {run_id} was started with prompt version {meta['prompt_version']}, "
                           f"current is {prompt_version}: use a new run_id")

    done = set(load_predictions(run_dir).get("id", pd.Series(dtype=str)))
    todo = rows[~rows["id"].astype(str).isin(done)]
    session = {"started": dt.datetime.now(dt.timezone.utc).isoformat(), "git_commit": git_commit(root),
               "model": client.s.model, "todo": len(todo), "usage_before": client.store.totals()}
    print(f"[{run_id}] {len(done)} already done · {len(todo)} to run")

    async def safe(row):
        try:
            return await item_fn(row)
        except (BudgetExceeded, DailyLimitReached):
            raise
        except Exception as e:  # noqa: BLE001  (API/network failure: logged, retried on resume)
            return {"id": str(row.id), "error": f"{type(e).__name__}: {str(e)[:300]}"}

    tasks = [asyncio.create_task(safe(r)) for r in todo.itertuples()]
    n_ok, errors, stop = 0, [], None
    with open(pred_path, "a", encoding="utf-8") as f:
        try:
            for i, fut in enumerate(asyncio.as_completed(tasks), 1):
                rec = await fut
                if "error" in rec:
                    errors.append(rec)
                    continue
                rec["run_id"] = run_id
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n_ok += 1
                if i % 25 == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)} complete · {len(errors)} errors")
        except (BudgetExceeded, DailyLimitReached) as e:
            stop = f"{type(e).__name__}: {e}"
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    err_path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in errors), encoding="utf-8")
    after = client.store.totals()
    session.update({"ended": dt.datetime.now(dt.timezone.utc).isoformat(), "completed": n_ok,
                    "errors": len(errors), "stopped": stop,
                    "usage_delta": {k: after[k] - session["usage_before"][k] for k in after}})
    del session["usage_before"]
    meta["sessions"].append(session)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    preds = load_predictions(run_dir)
    print(f"[{run_id}] this session: {n_ok} done, {len(errors)} errors"
          f"{' · STOPPED: ' + stop if stop else ''} · total predictions {len(preds)}/{len(rows)}")
    return preds