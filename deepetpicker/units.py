"""Pixel-size / diameter bookkeeping.

DeepETPicker works in voxels of whatever volume it is run on. Particle sizes
are known in Angstrom, so every conversion goes through here instead of being
done by hand in a GUI field.
"""
from __future__ import annotations

from dataclasses import dataclass

# DeepETPicker's README recommends binning so that the particle spans roughly
# 20-30 voxels.
RECOMMENDED_PARTICLE_VOXELS = (20, 30)

# The sample configs use a weak-label diameter of about half the real
# diameter (EMPIAR-10045: label 11 / ocp 23; SHREC2021: label 9 / ocp 7-17).
# This is a starting point, not a rule.
DEFAULT_LABEL_FRACTION = 0.5


@dataclass(frozen=True)
class DiameterSuggestion:
    pixel_size_A: float
    particle_voxels: int
    label_voxels: int
    in_recommended_range: bool


def binned_pixel_size(raw_pixel_size_A: float, binning: float) -> float:
    if raw_pixel_size_A <= 0 or binning <= 0:
        raise ValueError("pixel size and binning must be positive")
    return raw_pixel_size_A * binning


def suggest_diameters(pixel_size_A: float, particle_diameter_A: float,
                      label_fraction: float = DEFAULT_LABEL_FRACTION) -> DiameterSuggestion:
    """Particle ("ocp") and weak-label diameters in voxels at ``pixel_size_A``.

    Both are forced odd and >= 3 voxels, because the label template is centred
    on a voxel.
    """
    if pixel_size_A <= 0 or particle_diameter_A <= 0:
        raise ValueError("pixel size and diameter must be positive")
    ocp = _odd(particle_diameter_A / pixel_size_A)
    label = min(ocp, _odd(ocp * label_fraction))
    lo, hi = RECOMMENDED_PARTICLE_VOXELS
    return DiameterSuggestion(pixel_size_A, ocp, label, lo <= particle_diameter_A / pixel_size_A <= hi)


def _odd(x: float) -> int:
    n = max(3, int(round(x)))
    return n if n % 2 else n + 1


def coordinate_scale(from_pixel_size_A: float, to_pixel_size_A: float) -> float:
    """Factor to multiply coordinates by to go from one voxel grid to another.

    DeepETPicker voxels -> RELION tomogram pixels is
    ``coordinate_scale(deepet_px, relion_px)`` (same as the ``binning_factor``
    of RELION-US's DeepETPicker import bridge).
    """
    return from_pixel_size_A / to_pixel_size_A
