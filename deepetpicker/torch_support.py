"""Training-time helpers that need pytorch-lightning (imported lazily by train.py)."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytorch_lightning as pl


def _scalar(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class MetricsCSV(pl.Callback):
    """After every validation pass, append ``epoch`` plus every scalar metric
    the model logged to ``metrics.csv`` and print one ``epoch=… name=value``
    line (the Progress tab of RELION-US parses that). ``train_loss`` is the
    latest completed value, i.e. it can lag the validation metrics by an epoch."""

    def __init__(self, path):
        self.path = Path(path)
        self.fields = None

    def on_validation_end(self, trainer, pl_module):
        row = {"epoch": trainer.current_epoch}
        for k, v in sorted(trainer.callback_metrics.items()):
            s = _scalar(v)
            if s is not None:
                row[k] = s
        if self.fields is None:
            self.fields = list(row)
            new = not self.path.exists()
            self.f = open(self.path, "a", newline="")
            self.w = csv.DictWriter(self.f, fieldnames=self.fields, extrasaction="ignore")
            if new:
                self.w.writeheader()
        self.w.writerow(row)
        self.f.flush()
        print("epoch=%d " % row["epoch"] + " ".join(f"{k}={v:.6g}" for k, v in row.items() if k != "epoch"),
              flush=True)

    def on_train_end(self, trainer, pl_module):
        if getattr(self, "f", None):
            self.f.close()


def publish_best(checkpoint_callback, checkpoints_dir):
    """Copy the best checkpoint to ``best.ckpt`` (lightning names files per epoch)."""
    best = getattr(checkpoint_callback, "best_model_path", "")
    if best and Path(best).is_file():
        shutil.copyfile(best, Path(checkpoints_dir) / "best.ckpt")
        return True
    return False
