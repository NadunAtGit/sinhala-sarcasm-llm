"""Prompt rendering — Dev Plan v2 §6.1.

All prompt text lives in prompts/*.j2 (version-controlled, frozen at Gate G1).
`version()` hashes every template file plus few_shot.json, and is stored with each run,
so every prediction can be traced to the exact prompt text that produced it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

SCRIPT_LABEL = {"sinhala": "Sinhala script", "latin": "Latin script (Singlish or English)",
                "mixed": "mixed Sinhala and Latin script"}


def item_view(row, text_field: str = "text_enc", context_field: str = "context_enc") -> dict:
    """The fields of one comment that a prompt may show. Labels are included only for examples."""
    get = (lambda k: row[k]) if isinstance(row, dict) else (lambda k: getattr(row, k))
    ctx = (get(context_field) or "").strip()
    return {"id": str(get("id")), "text": get(text_field), "context": ctx or "(no caption)",
            "script_label": SCRIPT_LABEL.get(get("script_type"), get("script_type")),
            "is_sarcastic": int(get("is_sarcastic")), "sentiment": get("sentiment")}


class Prompts:
    def __init__(self, prompt_dir: str | Path):
        self.dir = Path(prompt_dir)
        self.env = Environment(loader=FileSystemLoader(str(self.dir)), undefined=StrictUndefined,
                               keep_trailing_newline=True, trim_blocks=False, lstrip_blocks=False)
        self._system = self.env.get_template("system.md.j2").render().strip()

    def system(self) -> str:
        return self._system

    def few_shot(self) -> list[dict]:
        path = self.dir / "few_shot.json"
        return json.loads(path.read_text(encoding="utf-8"))["examples"] if path.exists() else []

    def single_pass(self, item: dict, examples: list[dict] = (), note: str | None = None) -> list[dict]:
        user = self.env.get_template("single_pass.md.j2").render(item=item, examples=list(examples), note=note)
        return [{"role": "system", "content": self._system}, {"role": "user", "content": user.strip()}]

    def repair(self, error: str) -> str:
        return self.env.get_template("repair.md.j2").render(error=error[:150]).strip()

    def version(self) -> str:
        h = hashlib.sha256()
        for p in sorted(self.dir.rglob("*")):
            if p.is_file() and p.suffix in (".j2", ".json"):
                h.update(p.relative_to(self.dir).as_posix().encode())
                h.update(p.read_bytes())
        return h.hexdigest()[:12]