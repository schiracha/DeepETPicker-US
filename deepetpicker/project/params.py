"""``job_params.json`` and loading of the old ``pre_config=`` /
``train_configs=`` files."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1


def write_params(path, job_type: str, params: dict, inputs: dict | None = None) -> None:
    """Record everything that determines a job's output. ``inputs`` maps a
    role (e.g. ``dataset_star``) to a path, stored as given by the caller
    (use ``JobDir.relative`` so the file stays valid when the project moves)."""
    doc = {
        "schema_version": SCHEMA_VERSION,
        "job_type": job_type,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": inputs or {},
        "params": params,
    }
    Path(path).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


def read_params(path) -> dict:
    doc = json.loads(Path(path).read_text())
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema_version {doc.get('schema_version')!r}")
    return doc


def _unwrap(v):
    """The old GUI stored QFileDialog/Qt tuples in some fields; take the first."""
    return v[0] if isinstance(v, (list, tuple)) and len(v) >= 1 else v


def load_legacy_config(path) -> dict:
    """Load an old ``pre_config={...}`` or ``train_configs={...}`` file.

    The old code used ``text.lstrip('train_configs=')``, which strips a
    *character set*; the prefix is split off properly here.
    """
    text = Path(path).read_text()
    head, sep, body = text.partition("=")
    if not sep or not head.strip().isidentifier():
        raise ValueError(f"{path}: expected '<name>={{...}}', got {text[:40]!r}")
    cfg = json.loads(body)
    scalar_fields = {k: _unwrap(v) for k, v in cfg.items()}
    return scalar_fields
