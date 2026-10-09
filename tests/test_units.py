import pytest

from deepetpicker.units import binned_pixel_size, coordinate_scale, suggest_diameters


def test_bin3_at_3p12():
    px = binned_pixel_size(3.12, 3)
    assert px == pytest.approx(9.36)
    s = suggest_diameters(px, 150)
    assert s.particle_voxels == 17          # 16.0 voxels, forced odd
    assert s.label_voxels == 9
    assert not s.in_recommended_range       # README recommends 20-30 voxels


def test_bin3_at_2p1_is_in_range():
    s = suggest_diameters(binned_pixel_size(2.1, 3), 150)
    assert s.particle_voxels == 25 and s.in_recommended_range


def test_label_never_exceeds_particle():
    s = suggest_diameters(50, 150)
    assert 3 <= s.label_voxels <= s.particle_voxels


def test_coordinate_scale_to_relion():
    # picked on 6.3 A/px, RELION tomogram at 10.5 A/px
    assert coordinate_scale(6.3, 10.5) == pytest.approx(0.6)


def test_invalid():
    with pytest.raises(ValueError):
        suggest_diameters(0, 150)
