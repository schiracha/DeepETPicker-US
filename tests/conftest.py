import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def load_module(relpath):
    """Import ``utils/foo.py`` etc. by file path. ``import utils`` pulls in
    torch (utils/__init__.py), which the preprocessing code does not need."""
    path = REPO / relpath
    cached = sys.modules.get(path.stem)
    if cached is not None and getattr(cached, "__file__", None) == str(path):
        return cached      # a second copy would break pickling of its classes (multiprocessing.Pool)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def raw_data(tmp_path):
    from tests.synth import make_raw_data
    d = tmp_path / "raw_data"
    truth = make_raw_data(d)
    return d, truth
