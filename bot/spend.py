"""Append-only JSONL spend journal (AD-3 / CAP-3).

One line per agent call: ``{"ts", "chat_id", "cost_usd"}``. The cap check sums
the journal — it never estimates from tokens. File lives under gitignored
``journal/`` so a restart does not silently reset the cap mid-course
(bot-limits.md). Append-only: nothing is ever rewritten or deleted.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPEND_PATH = REPO_ROOT / "journal" / "spend.jsonl"


class SpendJournal:
    def __init__(self, path: Path | str = DEFAULT_SPEND_PATH) -> None:
        self.path = Path(path)

    def append(self, chat_id: int, cost_usd: float) -> None:
        """Persist one cost entry; call it after the agent returns/raises."""
        cost = float(cost_usd)
        if not math.isfinite(cost) or cost < 0:
            raise ValueError(f"cost_usd must be finite and non-negative, got {cost_usd!r}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "chat_id": chat_id,
            "cost_usd": cost,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def total(self) -> float:
        """Sum all journaled costs; corrupt or partial lines are skipped."""
        if not self.path.exists():
            return 0.0
        total = 0.0
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    total += float(json.loads(line)["cost_usd"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
        return total
