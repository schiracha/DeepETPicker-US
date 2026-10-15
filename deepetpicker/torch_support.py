"""Training-time helpers that need pytorch-lightning (imported lazily by train.py)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytorch_lightning as pl


def _scalar(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class MetricsCSV(pl.Callback):
    """After every training epoch, write ``epoch`` plus every scalar metric the
    model has logged to ``metrics.csv``, redraw ``loss_full.png`` and print one
    ``epoch=… name=value`` line.

    Uses ``on_train_epoch_end``, not the validation hook: validation runs before
    the epoch's ``train_loss`` is reduced (so a row written there carries the
    previous epoch's value) and also runs once as a sanity check before epoch 0,
    which would add a spurious row. The extra arguments absorb the signature
    differences between lightning versions."""

    def __init__(self, path):
        from .metrics import MetricsLog
        self.log = MetricsLog(path)

    def on_train_epoch_end(self, trainer, pl_module, *args, **kwargs):
        row = {"epoch": trainer.current_epoch}
        for k, v in sorted(trainer.callback_metrics.items()):
            s = _scalar(v)
            if s is not None:
                row[k] = s
        self.log.add(row)
        try:                                  # a failing plot must never stop a training run
            from .plots import plot_loss
            plot_loss(self.log.path, self.log.path.with_name("loss_full.png"))
        except Exception as e:                # noqa: BLE001
            print(f"warning: loss plot not updated: {e}", flush=True)
        print(self.log.line(row), flush=True)


def publish_best(checkpoint_callback, checkpoints_dir):
    """Copy the best checkpoint to ``best.ckpt`` (lightning names files per epoch)."""
    best = getattr(checkpoint_callback, "best_model_path", "")
    if best and Path(best).is_file():
        shutil.copyfile(best, Path(checkpoints_dir) / "best.ckpt")
        return True
    return False
