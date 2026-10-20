"""train.py unpacks seg_metrics() by position; the order must match what it returns.

Reading the source rather than importing it, because both modules need torch and a GPU.
The original code unpacked (iou, f1) as (f1, iou), so every logged ``val_f1`` was an IoU
and every ``val_iou`` was an F1.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _returned_names():
    tree = ast.parse((ROOT / "utils" / "metrics.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "seg_metrics")
    # first return is the isTrain one: mean_precision, mean_recall, mean_iou, mean_f1 (each ".item()")
    ret = next(n for n in ast.walk(fn) if isinstance(n, ast.Return))
    return [e.func.value.id for e in ret.value.elts]


def _unpacked_names():
    tree = ast.parse((ROOT / "train.py").read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "id", "") == "seg_metrics"):
            return [t.id for t in node.targets[0].elts]
    raise AssertionError("no `... = seg_metrics(...)` in train.py")


def _kind(name):
    name = name.lower()
    for key in ("precision", "recall", "iou", "f1"):
        if key in name:
            return key
    raise AssertionError(name)


def test_validation_unpacks_seg_metrics_in_return_order():
    assert [_kind(n) for n in _unpacked_names()] == [_kind(n) for n in _returned_names()]
    assert [_kind(n) for n in _returned_names()] == ["precision", "recall", "iou", "f1"]
