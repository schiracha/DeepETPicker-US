"""Loss-curve image for the Progress tab of RELION-US (``loss_full.png``).

RELION-US shows that file name for IsoNet2 training and, since the
DeepETPicker job types, for ``DeepETPickerTrain`` too. It is rewritten after
every validation pass, atomically, so a page that polls it never sees half a PNG.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

LOSS_COLUMNS = ("train_loss", "val_loss")


def plot_loss(metrics_csv, png_path) -> bool:
    """Plot the loss columns of ``metrics_csv``; False if there is nothing to plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(metrics_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    series = {c: [(int(r["epoch"]), float(r[c])) for r in rows if r.get(c) not in (None, "")]
              for c in LOSS_COLUMNS}
    series = {c: pts for c, pts in series.items() if pts}
    if not series:
        return False
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=110)
    for c, pts in series.items():
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3, label=c.replace("_", " "))
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    png_path = Path(png_path)
    tmp = png_path.with_name(png_path.name + ".tmp")
    fig.savefig(tmp, format="png")
    plt.close(fig)
    os.replace(tmp, png_path)
    return True
