# -*- coding: utf-8 -*-
"""Interactive max-detail cap for napari-imaris-loader.

Even with progressive scrubbing, the *settled* full-resolution refine is the
main residual cost when zoomed in: napari loads ~100-160 native tiles at the
finest level it deems necessary, each a latency-bound HDF5 read.  Often the
finest one or two pyramid levels are more detail than the screen can show
anyway.

This widget lets you cap how fine the settled view loads by dropping the
finest pyramid levels from each IMS layer in place and compensating the layer
scale so world coordinates are unchanged.  Capping at level 2-3 makes the
refine 4-16x cheaper.  It does *no* file I/O (it just re-points the layer at a
coarser slice of the already-loaded pyramid) and is fully reversible: set the
cap back to 0 for full detail.
"""

import napari
from magicgui import magic_factory
from qtpy.QtCore import QTimer
from napari.layers import Image

from ._logging import configure_logging, logger
from .progressive_loading_widget import _level_sequence, SCRUB_SUFFIX


# One controller per viewer, keyed by id(viewer).
_cap_controllers = {}


def reset_detail_cap(viewer):
    """Forget captured originals for a viewer (call after reloading data).

    A resolution reload recreates layers with the same names but a different
    pyramid, so any cached 'original' pyramid for those names is stale and must
    be dropped.
    """
    controller = _cap_controllers.get(id(viewer))
    if controller is not None:
        controller._timer.stop()
        controller._original.clear()
        controller._current.clear()


def _capped_scale(base_scale, base_shape, level_shape):
    """Scale for a layer whose finest level becomes ``level_shape``.

    Each dropped level halves resolution, so the new base level's voxel size
    is the original times the shape ratio per axis.  Keeping world extent
    (scale * shape) constant keeps the data aligned with any overlays.
    """
    out = []
    for i in range(len(base_scale)):
        bs = base_shape[i] if i < len(base_shape) else 1
        ls = level_shape[i] if i < len(level_shape) else 1
        out.append(base_scale[i] * (bs / ls if ls else 1.0))
    return tuple(out)


def _detect_resolution_levels(viewer):
    for layer in viewer.layers:
        levels = getattr(layer, 'metadata', {}).get('resolutionLevels')
        if levels:
            return int(levels)
    return None


class _DetailCapController:
    """Trims the finest pyramid levels from IMS layers (debounced, reversible)."""

    def __init__(self, viewer):
        self.viewer = viewer
        # parent name -> (full_data_list, base_scale_tuple) captured before any
        # cap so the original can always be restored / re-capped.
        self._original = {}
        # parent name -> currently applied cap level, to skip redundant work.
        self._current = {}
        self._pending = 0
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._apply_pending)

    def _ims_layers(self):
        for layer in self.viewer.layers:
            if not isinstance(layer, Image):
                continue
            if SCRUB_SUFFIX in layer.name:
                continue
            if not getattr(layer, 'metadata', {}).get('resolutionLevels'):
                continue
            yield layer

    def request(self, level, debounce_ms=250):
        """Debounce slider drags so we only re-point layers once per pause."""
        self._pending = int(level)
        self._timer.start(debounce_ms)

    def _apply_pending(self):
        self.apply(self._pending)

    def apply(self, level):
        configure_logging()
        applied = 0
        for layer in list(self._ims_layers()):
            if self._apply_to_layer(layer, level):
                applied += 1
        logger.info("detail_cap: applied level=%d to %d layer(s)", level, applied)

    def _apply_to_layer(self, layer, level):
        name = layer.name
        if name not in self._original:
            full = _level_sequence(layer.data)
            if full is None or len(full) < 2:
                return False
            self._original[name] = ([d for d in full], tuple(layer.scale))
        full_data, base_scale = self._original[name]
        n = len(full_data)
        # Always keep at least two levels so the layer stays multiscale.
        level = max(0, min(int(level), n - 2))
        if self._current.get(name) == level:
            return False  # already at this cap; avoid a redundant re-render
        new_data = full_data[level:]
        base_shape = full_data[0].shape
        lvl_shape = full_data[level].shape
        new_scale = _capped_scale(base_scale, base_shape, lvl_shape)
        try:
            layer.data = new_data
            layer.scale = new_scale
        except Exception as exc:
            logger.warning("detail_cap: failed to cap '%s': %s", name, exc)
            return False
        self._current[name] = level
        logger.info(
            "detail_cap: '%s' capped at level %d (kept %d level(s), base XY=%s)",
            name, level, len(new_data), tuple(lvl_shape[-2:]),
        )
        return True


def _detail_cap_init(widget):
    """Bound the slider to ``resolutionLevels - 2`` for the loaded file."""

    def _refresh(*_):
        try:
            viewer = widget.viewer.value
            if viewer is None:
                return
            levels = _detect_resolution_levels(viewer)
            if not levels:
                return
            slider = widget.max_detail_level
            slider.max = max(0, levels - 2)  # always leave >= 2 levels
            if slider.value > slider.max:
                slider.value = slider.max
        except Exception:
            pass

    def _hook_viewer(*_):
        viewer = widget.viewer.value
        if viewer is None:
            return
        if getattr(widget, '_imaris_cap_hooked', False):
            _refresh()
            return
        try:
            viewer.layers.events.inserted.connect(_refresh)
            viewer.layers.events.removed.connect(_refresh)
            widget._imaris_cap_hooked = True
        except Exception:
            pass
        _refresh()

    try:
        widget.viewer.changed.connect(_hook_viewer)
    except Exception:
        pass
    _hook_viewer()


@magic_factory(
    auto_call=True,
    widget_init=_detail_cap_init,
    max_detail_level={'widget_type': 'Slider', 'min': 0, 'max': 6,
                      'tooltip': 'Cap how finely the settled view loads. '
                                 '0 = full detail; higher = coarser but a much '
                                 'faster zoomed-in refine. No reload required.'},
)
def detail_cap(
    viewer: napari.Viewer,
    max_detail_level: int = 0,
):
    '''Cap the finest resolution the settled view loads (no reload).

    Drop the finest pyramid levels so zoomed-in refines read far fewer tiles.
    0 keeps full detail; raise it to trade detail for speed.  Reversible at any
    time and does no file I/O.
    '''
    configure_logging()
    controller = _cap_controllers.get(id(viewer))
    if controller is None:
        controller = _DetailCapController(viewer)
        _cap_controllers[id(viewer)] = controller
    controller.request(max_detail_level)
