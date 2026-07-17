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
slider is moving - or while the camera is zooming or panning - it shows the
cheap low-res companion and hides the expensive full-resolution layer; a short
while after you stop (debounced) it restores the full-resolution layer, which
then refines the view you landed on.

When the chosen companion level is small enough it is pulled fully into RAM, so
scrolling/zooming/panning triggers no HDF5 reads at all.

The full-resolution layer is only *hidden* during scrubbing (never removed), so
the slider range and world coordinates stay stable and no display settings need
to be synchronised.
"""

from collections.abc import Sequence
import time

import numpy as np

import napari
from magicgui import magic_factory

from qtpy.QtCore import QTimer

from napari.layers import Image

from ._logging import configure_logging, logger


def _level_sequence(data):
    """Return ``data`` as a sequence of pyramid levels, or ``None``.

    napari exposes a multiscale layer's ``data`` either as a plain ``list``
    (older versions / direct construction) or as a ``napari.layers``
    ``MultiScaleData`` object.  The latter is a ``collections.abc.Sequence``
    but is **not** a ``list``/``tuple`` subclass, so a naive
    ``isinstance(data, (list, tuple))`` check silently rejects every real
    multiscale layer.  Match on ``Sequence`` instead: ``list``/``tuple`` and
    ``MultiScaleData`` all qualify, while a bare numpy/dask array (a single
    plane) does not register as a ``Sequence`` and is correctly rejected.
    """
    if isinstance(data, (str, bytes)):
        return None
    if isinstance(data, Sequence):
        try:
            if len(data) >= 1:
                return data
        except TypeError:
            return None
    return None


SCRUB_SUFFIX = ' [scrub]'

# One controller per viewer, keyed by id(viewer).
_controllers = {}


class _ProgressiveController:
    """Wires dimension-scroll events to a low-res/high-res visibility swap."""

    # Pull a chosen scrub level fully into RAM when it is at most this many
    # bytes (per channel), so scrubbing/zooming never triggers HDF5 reads.
    # Larger levels stay lazy to avoid a long, blocking materialisation.
    RAM_BUDGET_BYTES = 512 * 2 ** 20

    def __init__(self, viewer, pause_delay_ms=300, scrub_max_pixels=1536):
        self.viewer = viewer
        self.pause_delay_ms = pause_delay_ms
        self.scrub_max_pixels = scrub_max_pixels
        self.active = False
        # Whether camera (zoom/pan) movement also triggers the low-res swap.
        self._camera_connected = False
        self._scrub_names = {}        # parent name -> scrub layer name
        self._parent_visible = {}     # parent name -> bool (visibility to restore)
        self._scrubbing = False
        # Background workers materialising scrub levels into RAM, and a build
        # generation counter so results that arrive after a rebuild/teardown
        # are ignored instead of overwriting fresh layers.
        self._workers = []
        self._build_gen = 0
        # Debounce restoring full-res after scrolling stops.
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_pause)
        # Debounce rebuilding the scrub layers when the quality slider changes,
        # so dragging the slider doesn't add/remove layers on every step.
        self._rebuild_timer = QTimer()
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.timeout.connect(self._apply_rebuild)
        # Debounce healing after layers are inserted (e.g. a resolution reload
        # replaces the IMS layers): build companions for the new layers once
        # the batch of insertions settles.
        self._heal_timer = QTimer()
        self._heal_timer.setSingleShot(True)
        self._heal_timer.timeout.connect(self._heal_inserted)
        self._layer_events_connected = False

    # -- construction of companion layers ---------------------------------
    @staticmethod
    def _choose_scrub_level(data_levels, max_pixels):
        """Smallest-index level whose XY fits in ``max_pixels``.

        Falls back to the coarsest available level when nothing is small
        enough (i.e. the minimum slider position gives the lowest resolution).
        """
        for i, arr in enumerate(data_levels):
            zyx = arr.shape[-3:]
            if max(zyx[-2:]) <= max_pixels:
                return i
        return len(data_levels) - 1

    def _schedule_ram_load(self, scrub_name, arr, parent_name, cache=None):
        """Materialise a small scrub level into RAM *off the UI thread*.

        The companion starts wrapping the lazy dask level (so it appears
        immediately), and a background worker computes the array and swaps it
        in when ready - turning the former ~1.7 s main-thread freeze on enable
        into a non-blocking refine.  Levels above ``RAM_BUDGET_BYTES`` stay
        lazy.

        ``cache`` (the shared chunk read cache, if any) has its writes paused
        for the duration of the bulk read so materialising this coarse level
        does not evict the hot fine-level chunks the cache exists to keep.
        """
        nbytes = getattr(arr, 'nbytes', None)
        if nbytes is not None and nbytes > self.RAM_BUDGET_BYTES:
            logger.info(
                "progressive: '%s' scrub kept lazy (%.0f MiB > %.0f MiB budget)",
                parent_name, nbytes / 2 ** 20, self.RAM_BUDGET_BYTES / 2 ** 20,
            )
            return

        gen = self._build_gen
        t0 = time.perf_counter()

        if cache is not None:
            cache.pause_writes()
        _resumed = {'done': False}

        def _resume():
            if cache is not None and not _resumed['done']:
                _resumed['done'] = True
                cache.resume_writes()

        def _done(mem):
            self._apply_loaded_scrub(scrub_name, mem, gen, parent_name,
                                     (time.perf_counter() - t0) * 1000)

        try:
            from napari.qt.threading import create_worker
        except Exception:
            # No Qt threading available: fall back to a (blocking) load so the
            # behaviour is still correct, just not backgrounded.
            try:
                _done(np.asarray(arr))
            except Exception as exc:
                logger.warning("progressive: '%s' RAM load failed: %s",
                               parent_name, exc)
            finally:
                _resume()
            return

        worker = create_worker(np.asarray, arr)
        worker.returned.connect(_done)
        worker.errored.connect(
            lambda exc, pn=parent_name: logger.warning(
                "progressive: '%s' background RAM load failed, staying lazy: %s",
                pn, exc)
        )
        # Hold a reference so the worker isn't garbage-collected mid-flight;
        # resume cache writes and drop the reference when finished (success or
        # error).
        self._workers.append(worker)

        def _finished(w=worker):
            _resume()
            if w in self._workers:
                self._workers.remove(w)

        worker.finished.connect(_finished)
        worker.start()

    def _apply_loaded_scrub(self, scrub_name, mem, gen, parent_name, elapsed_ms):
        """Swap a freshly materialised array into its scrub layer (UI thread)."""
        if gen != self._build_gen:
            return  # layers were rebuilt/torn down while loading -> discard
        if scrub_name not in self.viewer.layers:
            return
        try:
            self.viewer.layers[scrub_name].data = mem
        except Exception as exc:
            logger.warning("progressive: '%s' could not swap in RAM scrub: %s",
                           parent_name, exc)
            return
        logger.info(
            "progressive: '%s' scrub loaded into RAM (%.0f MiB) in %.0f ms "
            "(background)",
            parent_name, getattr(mem, 'nbytes', 0) / 2 ** 20, elapsed_ms,
        )

    def _make_scrub_layer(self, parent):
        is_multiscale = getattr(parent, 'multiscale', None)
        # ``layer.data`` is a sequence of pyramid arrays for a multiscale layer
        # (a ``list`` or a ``MultiScaleData`` sequence) and a single array
        # otherwise.  Log exactly which case we hit so a session that reports
        # "no multiscale layers to accelerate" can be diagnosed from the log.
        data_levels = _level_sequence(parent.data)
        if data_levels is None:
            logger.info(
                "progressive: skip '%s' - data is %s (multiscale=%s); "
                "expected a sequence of pyramid levels",
                parent.name, type(parent.data).__name__, is_multiscale,
            )
            return None  # not multiscale -> scrubbing wouldn't help
        if len(data_levels) < 2:
            logger.info(
                "progressive: skip '%s' - only %d resolution level(s) "
                "(multiscale=%s); nothing coarser to scrub with",
                parent.name, len(data_levels), is_multiscale,
            )
            return None

        level = self._choose_scrub_level(data_levels, self.scrub_max_pixels)
        if level == 0:
            logger.info(
                "progressive: skip '%s' - level 0 already fits scrub_max_pixels"
                "=%d (XY=%s); nothing to gain",
                parent.name, self.scrub_max_pixels,
                tuple(data_levels[0].shape[-2:]),
            )
            return None  # already cheap enough; nothing to gain

        coarse = data_levels[level]
        # Use the last min(3, ndim) spatial dims so 2D (y,x) layers work too;
        # hard-coding 3 IndexErrors on 2D multiscale data.
        ndim = min(3, len(parent.scale))
        full_shape = data_levels[0].shape[-ndim:]
        coarse_shape = coarse.shape[-ndim:]
        pscale = tuple(parent.scale[-ndim:])
        sscale = tuple(
            pscale[d] * full_shape[d] / coarse_shape[d] for d in range(ndim)
        )

        # Replacing an existing companion of the same name (e.g. after a
        # rebuild) avoids napari auto-suffixing duplicates ('... [scrub] [1]').
        scrub_name = parent.name + SCRUB_SUFFIX
        if scrub_name in self.viewer.layers:
            try:
                self.viewer.layers.remove(scrub_name)
            except Exception:
                pass

        # Add immediately with the lazy level, then materialise into RAM in the
        # background and swap it in (see _schedule_ram_load).
        scrub = self.viewer.add_image(
            coarse,
            name=scrub_name,
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
        xy_factor = full_shape[-1] / coarse_shape[-1]
        logger.info(
            "progressive: '%s' scrub uses level %d of %d, shape=%s "
            "(~%.0fx downsampled in XY, scrub_max_pixels=%d)",
            parent.name, level, len(data_levels) - 1, coarse_shape,
            xy_factor, self.scrub_max_pixels,
        )
        parent_meta = getattr(parent, 'metadata', None) or {}
        cache = parent_meta.get('_ims_read_cache')
        self._schedule_ram_load(scrub_name, coarse, parent.name, cache=cache)
        return scrub

    def _build_scrub_layers(self):
        candidates = 0
        built = 0
        for layer in list(self.viewer.layers):
            if not isinstance(layer, Image):
                continue
            # ``in`` (not ``endswith``) so an auto-suffixed orphan like
            # 'Ch0 [scrub] [1]' is treated as a companion, never a parent.
            if SCRUB_SUFFIX in layer.name:
                continue
            # Skip parents that already have a live companion (so healing after
            # an insertion only builds for genuinely new layers).
            existing = self._scrub_names.get(layer.name)
            if existing is not None and existing in self.viewer.layers:
                continue
            candidates += 1
            try:
                scrub = self._make_scrub_layer(layer)
            except Exception as exc:
                logger.warning("progressive: could not build scrub layer for "
                               "'%s': %s", layer.name, exc)
                scrub = None
            if scrub is not None:
                self._scrub_names[layer.name] = scrub.name
                built += 1
        logger.info(
            "progressive: inspected %d new image layer(s), built %d scrub "
            "layer(s) (%d tracked total)",
            candidates, built, len(self._scrub_names),
        )

    def _remove_scrub_layers(self):
        self._build_gen += 1  # discard results from any in-flight RAM loads
        # If we were mid-scrub, restore originals before tearing down.
        if self._scrubbing:
            self._restore_full_res()
        for _parent_name, scrub_name in list(self._scrub_names.items()):
            if scrub_name in self.viewer.layers:
                try:
                    self.viewer.layers.remove(scrub_name)
                except Exception:
                    pass
        self._scrub_names.clear()
        self._parent_visible.clear()
        self._scrubbing = False

    # -- enable / disable -------------------------------------------------
    def enable(self, scrub_max_pixels):
        configure_logging()
        if self.active:
            # Already running: only rebuild (debounced) if the quality changed.
            if scrub_max_pixels != self.scrub_max_pixels:
                self.scrub_max_pixels = scrub_max_pixels
                self._rebuild_timer.start(300)
            return

        self.scrub_max_pixels = scrub_max_pixels
        self._build_scrub_layers()
        if not self._scrub_names:
            logger.info("progressive: no multiscale layers to accelerate")
            return

        self.viewer.dims.events.current_step.connect(self._on_interact)
        self._connect_camera()
        self._connect_layer_events()
        self.active = True
        logger.info("progressive loading enabled (%d layer(s), delay=%d ms, "
                    "scrub_max_pixels=%d, zoom/pan accelerated)",
                    len(self._scrub_names), self.pause_delay_ms,
                    self.scrub_max_pixels)

    def _connect_layer_events(self):
        """Self-heal when layers are added/removed (e.g. a resolution reload).

        Rebuilding companions for freshly inserted layers keeps the toggle live
        across a reload, and dropping companions whose parent was removed avoids
        stale overlays - without the resolution widget having to reach in and
        purge.
        """
        if self._layer_events_connected:
            return
        try:
            self.viewer.layers.events.inserted.connect(self._on_layers_inserted)
            self.viewer.layers.events.removed.connect(self._on_layers_removed)
            self._layer_events_connected = True
        except Exception as exc:
            logger.warning("progressive: could not hook layer events "
                           "(no self-heal on reload): %s", exc)

    def _disconnect_layer_events(self):
        if not self._layer_events_connected:
            return
        try:
            self.viewer.layers.events.inserted.disconnect(self._on_layers_inserted)
        except Exception:
            pass
        try:
            self.viewer.layers.events.removed.disconnect(self._on_layers_removed)
        except Exception:
            pass
        self._layer_events_connected = False

    def _on_layers_inserted(self, event=None):
        # Debounce so a batch of insertions heals once, off the event stack.
        if self.active:
            self._heal_timer.start(200)

    def _heal_inserted(self):
        if not self.active:
            return
        before = len(self._scrub_names)
        self._build_scrub_layers()
        if len(self._scrub_names) != before:
            logger.info("progressive: healed after layer insertion "
                        "(%d companion(s) tracked)", len(self._scrub_names))

    def _on_layers_removed(self, event=None):
        removed = getattr(event, 'value', None)
        name = getattr(removed, 'name', None)
        if name is None:
            return
        # A tracked parent went away: drop and remove its companion.
        if name in self._scrub_names:
            scrub_name = self._scrub_names.pop(name)
            self._parent_visible.pop(name, None)
            if scrub_name in self.viewer.layers:
                try:
                    self.viewer.layers.remove(scrub_name)
                except Exception:
                    pass
            return
        # A companion was removed directly: forget its tracking entry.
        for pname, sname in list(self._scrub_names.items()):
            if sname == name:
                self._scrub_names.pop(pname, None)
                self._parent_visible.pop(pname, None)

    def _connect_camera(self):
        """Swap to low-res while the camera zooms or pans, not only on scroll."""
        if self._camera_connected:
            return
        try:
            self.viewer.camera.events.zoom.connect(self._on_interact)
            self.viewer.camera.events.center.connect(self._on_interact)
            self._camera_connected = True
        except Exception as exc:
            logger.warning("progressive: could not hook camera events "
                           "(zoom/pan acceleration disabled): %s", exc)

    def _disconnect_camera(self):
        if not self._camera_connected:
            return
        for evt in (self.viewer.camera.events.zoom,
                    self.viewer.camera.events.center):
            try:
                evt.disconnect(self._on_interact)
            except Exception:
                pass
        self._camera_connected = False

    def _apply_rebuild(self):
        if not self.active:
            return
        self._remove_scrub_layers()
        self._build_scrub_layers()
        logger.info("progressive: rebuilt scrub layers "
                    "(scrub_max_pixels=%d)", self.scrub_max_pixels)

    def disable(self):
        if self.active:
            try:
                self.viewer.dims.events.current_step.disconnect(self._on_interact)
            except Exception:
                pass
            self._disconnect_camera()
        self._disconnect_layer_events()
        self._timer.stop()
        self._rebuild_timer.stop()
        self._heal_timer.stop()
        self._remove_scrub_layers()
        self.active = False

    # -- interaction handling (z-scroll + camera zoom/pan) ----------------
    def _on_interact(self, event=None):
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
                        'zooming or panning, then refine to full resolution '
                        'when you pause.'},
    pause_delay_ms={'widget_type': 'Slider', 'min': 50, 'max': 2000, 'step': 50,
                    'tooltip': 'How long after you stop interacting before the '
                               'full-resolution image is restored.'},
    scrub_max_pixels={'widget_type': 'Slider', 'min': 128, 'max': 8192,
                      'step': 128,
                      'tooltip': 'Largest XY size (pixels) allowed for the '
                                 'low-resolution scrubbing image. Drag left for '
                                 'a lower-resolution (faster, blockier) scrub; '
                                 'the minimum uses the coarsest pyramid level.'},
)
def progressive_loading(
    viewer: napari.Viewer,
    enabled: bool = False,
    pause_delay_ms: int = 300,
    scrub_max_pixels: int = 1536,
):
    '''Dynamically drop to a low resolution while scrolling, zooming or panning.

    Enable this, then scroll through Z (or any slider dimension), zoom or pan.
    While you interact you see a fast low-resolution image; shortly after you
    stop, the full-resolution plane is loaded.  The low-resolution image is
    held in RAM (when small enough) so interaction triggers no disk reads.
    Drag *scrub max pixels* to the left for an even lower-resolution scrub
    image.  Toggle off (or re-toggle after reloading data) to remove the
    helper layers.
    '''
    configure_logging()
    controller = _controllers.get(id(viewer))
    if controller is None:
        controller = _ProgressiveController(viewer)
        _controllers[id(viewer)] = controller

    controller.pause_delay_ms = pause_delay_ms

    if enabled:
        controller.enable(scrub_max_pixels)
    else:
        controller.disable()
