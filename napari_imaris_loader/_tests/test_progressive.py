"""Tests for the progressive (scrub) loading helpers."""

from collections.abc import Sequence

import numpy as np

from napari_imaris_loader.progressive_loading_widget import _level_sequence


class _FakeMultiScaleData(Sequence):
    """Stand-in for napari's ``MultiScaleData``.

    Like the real thing it is a ``collections.abc.Sequence`` (not a ``list``/
    ``tuple`` subclass) and it exposes a ``shape`` property mirroring level 0.
    Both traits previously broke scrub-layer detection.
    """

    def __init__(self, levels):
        self._levels = list(levels)

    def __getitem__(self, idx):
        return self._levels[idx]

    def __len__(self):
        return len(self._levels)

    @property
    def shape(self):
        return self._levels[0].shape


def test_level_sequence_accepts_list():
    levels = [np.zeros((2, 8, 8)), np.zeros((2, 4, 4))]
    assert _level_sequence(levels) is levels


def test_level_sequence_accepts_tuple():
    levels = (np.zeros((2, 8, 8)), np.zeros((2, 4, 4)))
    assert _level_sequence(levels) is levels


def test_level_sequence_accepts_multiscale_data():
    """Regression: a MultiScaleData-style Sequence must be recognised.

    The old ``isinstance(data, (list, tuple))`` gate rejected this object, so
    no scrub layer was ever built and the widget logged
    "no multiscale layers to accelerate".
    """
    msd = _FakeMultiScaleData([np.zeros((2, 8, 8)), np.zeros((2, 4, 4))])
    assert _level_sequence(msd) is msd
    assert len(_level_sequence(msd)) == 2


def test_level_sequence_rejects_single_array():
    # A bare numpy array is a single plane, not a pyramid.
    assert _level_sequence(np.zeros((2, 8, 8))) is None


def test_level_sequence_rejects_str_and_empty():
    assert _level_sequence("not-data") is None
    assert _level_sequence([]) is None
