"""Weak-label and occupancy volumes from point coordinates.

This is a line-for-line port of ``Coord_to_Label.single_handle`` in
``utils/coords2labels.py`` (so training data are unchanged), with three
differences: it takes arrays instead of reading ``base_path/coords``, the
output is written through a memory map so the volume is never held in RAM,
and even diameters are rejected up front (the old code crashed on them with a
broadcasting error, because the template has ``dim`` voxels but the window
``2*int(dim/2)+1``).

Behaviour kept on purpose, because trained models depend on it: where a new
particle's window overlaps an earlier label, every non-zero voxel in the
window takes the *new* particle's value.
"""
from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np

LABEL_TYPES = ("gaussian", "sphere", "cubic")  # as offered by the old GUI
OCP = "data_ocp"


def gaussian3d(shape, sigma=1):
    l, m, n = [(ss - 1.0) / 2.0 for ss in shape]
    z, y, x = np.ogrid[-l:l + 1, -m:m + 1, -n:n + 1]
    sigma = (sigma - 1.0) / 2.0
    return np.exp(-(x * x + y * y + z * z) / (2 * sigma * sigma))


def _threshold(label_type):
    t = label_type.lower()
    if "cubic" in t:
        return 0.223        # exp(-1.5)
    if "sphere" in t or "ocp" in t:
        return 0.60653      # exp(-0.5)
    return 0.367879         # exp(-1)


def _check_diameters(diameters):
    for d in diameters:
        if int(d) < 3 or int(d) % 2 == 0:
            raise ValueError(f"diameters must be odd and >= 3 voxels, got {d}")


def fill_labels(label, coords, label_type, diameters, num_cls):
    """Paint ``coords`` ((N, 4) ``class x y z`` integer voxels) into ``label``
    (a 3-D array, ``z, y, x``) in place."""
    diameters = [int(d) for d in diameters]
    _check_diameters(diameters)
    z_max, y_max, x_max = label.shape
    is_ocp = OCP in label_type.lower()
    tg = _threshold(label_type)
    for pos_idx, (cls_idx_, x, y, z) in enumerate(np.asarray(coords, dtype=np.int64)):
        dim = diameters[cls_idx_ - 1] if is_ocp else diameters[0]
        r = int(dim / 2)
        template = gaussian3d((dim, dim, dim), dim)
        cls_idx = pos_idx + 1 if is_ocp else cls_idx_

        z_start = max(z - r, 0); z_end = min(z + r + 1, z_max)
        y_start = max(y - r, 0); y_end = min(y + r + 1, y_max)
        x_start = max(x - r, 0); x_end = min(x + r + 1, x_max)
        t_z_start = r - z if z - r < 0 else 0
        t_z_end = (r + z_max - z) if z + r + 1 > z_max else 2 * r + 1
        t_y_start = r - y if y - r < 0 else 0
        t_y_end = (r + y_max - y) if y + r + 1 > y_max else 2 * r + 1
        t_x_start = r - x if x - r < 0 else 0
        t_x_end = (r + x_max - x) if x + r + 1 > x_max else 2 * r + 1

        tmp1 = label[z_start:z_end, y_start:y_end, x_start:x_end]
        tmp2 = template[t_z_start:t_z_end, t_y_start:t_y_end, t_x_start:t_x_end]
        larger = tmp1 < tmp2
        tmp1[larger] = tmp2[larger]
        tmp1[tmp1 <= tg] = 0
        label[z_start:z_end, y_start:y_end, x_start:x_end] = np.where(tmp1 > 0, cls_idx, 0)


def write_label_file(dst, tomo_file, coords, label_type, diameters, num_cls):
    """Create ``dst`` with the shape and voxel size of ``tomo_file``."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.mmap(tomo_file, mode="r", permissive=True) as t:
        shape, vs = t.data.shape, t.voxel_size.copy()
    with mrcfile.new_mmap(dst, shape=shape, mrc_mode=2, overwrite=True) as out:
        out.data[...] = 0
        fill_labels(out.data, coords, label_type, diameters, num_cls)
        out.voxel_size = vs
