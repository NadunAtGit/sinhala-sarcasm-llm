"""Convert thesis splits to data/processed/*.jsonl, run §5.1 checks, write eval-set ID lists.

    python scripts/prepare_data.py --map configs/data_map.yaml
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
import yaml

from sarcllm.data import (
    build_option_b, build_pilot, check_counts, detect_script, load_all, write_ids, write_jsonl,
)

app = typer.Typer(add_completion=False)


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype=str)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True, dtype=str)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _map_values(series: pd.Series, mapping: dict, name: str) -> pd.Series:
    m = {str(k).strip(): v for k, v in mapping.items()}
    out = series.astype(str).str.strip().map(m)
    bad = series[out.isna()].unique()
    if len(bad):
        raise typer.BadParameter(f"unmapped {name} values: {list(bad)[:10]} — add them to data_map.yaml")
    return out


def convert(split: str, cfg: dict, base: Path) -> pd.DataFrame:
    raw = _read_any((base / cfg["inputs"][split]).resolve())
    c = cfg["columns"]
    df = pd.DataFrame()
    df["id"] = (raw[c["id"]].astype(str) if c.get("id")
                else cfg["id_prefix"][split] + raw.index.astype(str))
    df["post_id"] = raw[c["post_id"]].astype(str)
    df["text"] = raw[c["text"]].fillna("").astype(str)
    df["context_text"] = raw[c["context_text"]].fillna("").astype(str) if c.get("context_text") else ""
    df["sentiment"] = _map_values(raw[c["sentiment"]], cfg["sentiment_values"], "sentiment")
    df["is_sarcastic"] = _map_values(raw[c["is_sarcastic"]], cfg["sarcasm_values"], "sarcasm").astype(int)
    df["script_type"] = (raw[c["script_type"]].str.lower().str.strip() if c.get("script_type")
                         else df["text"].map(detect_script))
    df["split"] = split
    return df


@app.command()
def main(
    map: Path = typer.Option(Path("configs/data_map.yaml")),
    out_dir: Path = typer.Option(Path("data/processed")),
    eval_dir: Path = typer.Option(Path("data/eval_sets")),
    strict: bool = typer.Option(True, help="Exit non-zero if any §5.1 count fails"),
):
    cfg = yaml.safe_load(map.read_text(encoding="utf-8"))
    base = map.parent
    for split in ("train", "val", "test"):
        write_jsonl(convert(split, cfg, base), out_dir / f"{split}.jsonl")

    splits = load_all(out_dir)  # re-load through the pydantic validator
    errs = check_counts(splits)
    hard = [e for e in errs if not e.startswith("WARNING")]
    for e in errs:
        typer.echo(("  ! " if e in hard else "  ~ ") + e)
    if not errs:
        typer.echo("§5.1 count checks: all pass")

    te, va = splits["test"], splits["val"]
    typer.echo("\nscript distribution (test):\n" + te.script_type.value_counts().to_string())

    write_ids(sorted(te.id), eval_dir / "test_full.txt")
    write_ids(build_option_b(te, seed=42), eval_dir / "test_optB.txt")
    try:
        write_ids(build_pilot(va, seed=42), eval_dir / "pilot_val_300.txt")
    except ValueError as e:
        typer.echo(f"  ! pilot not built: {e}")
    typer.echo(f"eval-set ID lists written to {eval_dir} — commit them")

    if strict and hard:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
