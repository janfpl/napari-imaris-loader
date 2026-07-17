"""Tests for the detail-cap scale math."""

from napari_imaris_loader.detail_cap_widget import _capped_scale


def test_capped_scale_restore_is_identity():
    base = (2.0, 0.65, 0.65)
    shape = (1597, 21432, 8524)
    assert _capped_scale(base, shape, shape) == base


def test_capped_scale_doubles_per_level():
    base = (2.0, 0.65, 0.65)
    l0 = (1597, 21432, 8524)
    l1 = (798, 10716, 4262)
    z, y, x = _capped_scale(base, l0, l1)
    # Each axis grows by the shape ratio (~2x per dropped level).
    assert abs(z - 2.0 * 1597 / 798) < 1e-6
    assert abs(y - 0.65 * 21432 / 10716) < 1e-6
    assert abs(x - 0.65 * 8524 / 4262) < 1e-6


def test_capped_scale_keeps_world_extent_constant():
    # scale * shape (the physical extent) must be preserved when capping.
    base = (2.0, 0.65, 0.65)
    l0 = (1597, 21432, 8524)
    l2 = (399, 5358, 2131)
    capped = _capped_scale(base, l0, l2)
    for i in range(3):
        assert abs(capped[i] * l2[i] - base[i] * l0[i]) < 1e-3
