from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = {"timestamp": datetime.now().isoformat(timespec="seconds"), **payload}
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(enriched, ensure_ascii=False) + "\n")


def read_jsonl(path: Path, limit: int = 100) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    return rows[-limit:]
