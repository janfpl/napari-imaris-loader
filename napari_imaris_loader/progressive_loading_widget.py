# -*- coding: utf-8 -*-
"""Progressive (scrub) loading for napari-imaris-loader.

Profiling large multicolor IMS files showed that reading data is *not* the
bottleneck while scrolling through full-resolution planes: a whole plane's
worth of tiles comes back in ~0.3 s with good parallelism.  The cost is paying
napari's full-resolution slice + texture build on *every* intermediate plane as
you scrub.

This widget avoids that.  For every multiscale image layer it creates a hidden,
single low-resolution "scrub" companion layer that overlays the original
exactly (matching scale/translate/colormap/contrast).  While the dimension
slider is moving it shows the cheap low-res companion and hides the expensive
full-resolution layer; a short while after you stop (debounced) it restores the
full-resolution layer, which then refines the plane you landed on.

The full-resolution layer is only *hidden* during scrubbing (never removed), so
the slider range and world coordinates stay stable and no display settings need
to be synchronised.
"""

import napari
from magicgui import magic_factory

from qtpy.QtCore import QTimer

from napari.layers import Image

from ._logging import configure_logging, logger


SCRUB_SUFFIX = ' [scrub]'

# One controller per viewer, keyed by id(viewer).
_controllers = {}


class _ProgressiveController:
    """Wires dimension-scroll events to a low-res/high-res visibility swap."""

    def __init__(self, viewer, pause_delay_ms=300, scrub_max_pixels=1536):
        self.viewer = viewer
        self.pause_delay_ms = pause_delay_ms
        self.scrub_max_pixels = scrub_max_pixels
        self.active = False
        self._scrub_names = {}        # parent name -> scrub layer name
        self._parent_visible = {}     # parent name -> bool (visibility to restore)
        self._scrubbing = False
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_pause)

    # -- construction of companion layers ---------------------------------
    @staticmethod
    def _choose_scrub_level(data_levels, max_pixels):
        for i, arr in enumerate(data_levels):
            zyx = arr.shape[-3:]
            if max(zyx[-2:]) <= max_pixels:
                return i
        return len(data_levels) - 1

    def _make_scrub_layer(self, parent):
        data_levels = parent.data
        if not isinstance(data_levels, (list, tuple)) or len(data_levels) < 2:
            return None  # not multiscale -> scrubbing wouldn't help

        level = self._choose_scrub_level(data_levels, self.scrub_max_pixels)
        if level == 0:
            return None  # already cheap enough; nothing to gain

        coarse = data_levels[level]
        full_shape = data_levels[0].shape[-3:]
        coarse_shape = coarse.shape[-3:]
        pscale = tuple(parent.scale[-3:])
        sscale = tuple(
            pscale[d] * full_shape[d] / coarse_shape[d] for d in range(3)
        )

        scrub = self.viewer.add_image(
            coarse,
            name=parent.name + SCRUB_SUFFIX,
            scale=sscale,
            translate=tuple(parent.translate[-3:]),
            colormap=parent.colormap,
            blending=parent.blending,
            contrast_limits=parent.contrast_limits,
            gamma=parent.gamma,
            rendering=parent.rendering,
            opacity=parent.opacity,
            multiscale=False,
            visible=False,
        )
        logger.info(
            "progressive: scrub layer for '%s' uses level %d shape=%s",
            parent.name, level, coarse_shape,
        )
        return scrub

    # -- enable / disable -------------------------------------------------
    def enable(self):
        # Rebuild cleanly so re-toggling picks up the current layers/settings.
        self.disable()
        configure_logging()

        for layer in list(self.viewer.layers):
            if not isinstance(layer, Image):
                continue
            if layer.name.endswith(SCRUB_SUFFIX):
                continue
            try:
                scrub = self._make_scrub_layer(layer)
            except Exception as exc:
                logger.warning("progressive: could not build scrub layer for "
                               "'%s': %s", layer.name, exc)
                scrub = None
            if scrub is not None:
                self._scrub_names[layer.name] = scrub.name

        if not self._scrub_names:
            logger.info("progressive: no multiscale layers to accelerate")
            return

        self.viewer.dims.events.current_step.connect(self._on_scroll)
        self.active = True
        logger.info("progressive loading enabled (%d layer(s), delay=%d ms)",
                    len(self._scrub_names), self.pause_delay_ms)

    def disable(self):
        if self.active:
            try:
                self.viewer.dims.events.current_step.disconnect(self._on_scroll)
            except Exception:
                pass
        self._timer.stop()

        # If we were mid-scrub, restore originals before tearing down.
        if self._scrubbing:
            self._restore_full_res()

        for parent_name, scrub_name in list(self._scrub_names.items()):
            if scrub_name in self.viewer.layers:
                try:
                    self.viewer.layers.remove(scrub_name)
                except Exception:
                    pass
        self._scrub_names.clear()
        self._parent_visible.clear()
        self._scrubbing = False
        self.active = False

    # -- scroll handling --------------------------------------------------
    def _on_scroll(self, event=None):
        if not self._scrub_names:
            return
        if not self._scrubbing:
            self._scrubbing = True
            for parent_name, scrub_name in self._scrub_names.items():
                if parent_name not in self.viewer.layers or scrub_name not in self.viewer.layers:
                    continue
                parent = self.viewer.layers[parent_name]
                scrub = self.viewer.layers[scrub_name]
                was_visible = parent.visible
                self._parent_visible[parent_name] = was_visible
                scrub.visible = was_visible
                parent.visible = False
        # Restart the debounce: full-res restored only after scrolling pauses.
        self._timer.start(self.pause_delay_ms)

    def _on_pause(self):
        self._restore_full_res()

    def _restore_full_res(self):
        for parent_name, scrub_name in self._scrub_names.items():
            if parent_name in self.viewer.layers:
                self.viewer.layers[parent_name].visible = (
                    self._parent_visible.get(parent_name, True)
                )
            if scrub_name in self.viewer.layers:
                self.viewer.layers[scrub_name].visible = False
        self._scrubbing = False


@magic_factory(
    auto_call=True,
    enabled={'tooltip': 'Show a fast low-resolution image while scrolling, '
                        'then refine to full resolution when you pause.'},
    pause_delay_ms={'min': 50, 'max': 2000, 'step': 50,
                    'tooltip': 'How long after scrolling stops before the '
                               'full-resolution image is restored.'},
    scrub_max_pixels={'min': 128, 'max': 8192, 'step': 128,
                      'tooltip': 'Largest XY size (pixels) allowed for the '
                                 'low-resolution scrubbing image. Smaller = '
                                 'faster scrolling, blurrier while moving.'},
)
def progressive_loading(
    viewer: napari.Viewer,
    enabled: bool = False,
    pause_delay_ms: int = 300,
    scrub_max_pixels: int = 1536,
):
    '''Dynamically drop to a low resolution while scrolling planes.

    Enable this, then scroll through Z (or any slider dimension) at full zoom.
    While the slider moves you see a fast low-resolution image; shortly after
    you stop, the full-resolution plane is loaded.  Toggle off (or re-toggle
    after reloading data) to remove the helper layers.
    '''
    configure_logging()
    controller = _controllers.get(id(viewer))
    if controller is None:
        controller = _ProgressiveController(viewer)
        _controllers[id(viewer)] = controller

    controller.pause_delay_ms = pause_delay_ms
    controller.scrub_max_pixels = scrub_max_pixels

    if enabled:
        controller.enable()
    else:
        controller.disable()
