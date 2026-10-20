"""``metrics.csv``: one row per epoch, columns added as new metrics appear.

Kept free of PyTorch so it can be tested without a GPU; the lightning callback
in ``torch_support`` only collects the numbers.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path


class MetricsLog:
    """Rewrites the whole (small) file after every row, so the header always
    covers every column seen so far. Appending under a header fixed by the first
    row would silently drop a metric that only shows up from the second epoch on."""

    def __init__(self, path):
        self.path = Path(path)
        self.rows: list[dict] = []
        self.fields: list[str] = ["epoch"]

    def add(self, row: dict) -> None:
        for k in row:
            if k not in self.fields:
                self.fields.append(k)
        self.rows.append(dict(row))
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fields, restval="")
            w.writeheader()
            w.writerows(self.rows)
        os.replace(tmp, self.path)

    @staticmethod
    def line(row: dict) -> str:
        """The progress line, starting on a fresh line (progress bars end with a
        carriage return, which would otherwise glue it to the bar)."""
        return "\nepoch=%d " % row["epoch"] + " ".join(f"{k}={v:.6g}" for k, v in row.items() if k != "epoch")
