# -*- coding: utf-8 -*-
"""Hidden, opt-in verbose logging for napari-imaris-loader.

This is a *troubleshooting* facility.  It is completely silent and adds no
measurable overhead unless it is explicitly switched on with an environment
variable, so it is safe to leave in the shipped plugin.

Enable it BEFORE launching napari:

    Windows (cmd):     set NAPARI_IMARIS_LOG=DEBUG
    Windows (PowerShell): $env:NAPARI_IMARIS_LOG="DEBUG"
    macOS / Linux:     export NAPARI_IMARIS_LOG=DEBUG

Accepted values are standard logging levels (DEBUG, INFO, WARNING ...).
``DEBUG`` is the interesting one: it logs the duration and shape of every
chunk that napari/dask asks the reader to load, which is what reveals where
time is actually being spent while you interact with the GUI.

Optionally send the log to a file.  This is strongly recommended when chasing
a GUI freeze, because the records are flushed immediately and therefore
survive even if the viewer becomes unresponsive:

    set NAPARI_IMARIS_LOG_FILE=C:\\temp\\imaris_loader.log

Reading the output
------------------
* ``read level=0 key=(...) -> shape=... in 1234.5 ms`` lines come from actual
  HDF5 reads.  If these are slow, the bottleneck is disk / decompression /
  chunking.
* If those lines are fast (or absent) but the GUI is still unresponsive, the
  time is being spent inside napari / vispy rendering, not in this plugin.
"""

import logging
import os
import sys
import time
from contextlib import contextmanager

logger = logging.getLogger("napari_imaris_loader")

_CONFIGURED = False


def configure_logging():
    """Configure the package logger from environment variables (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return logger
    _CONFIGURED = True

    level_name = os.environ.get("NAPARI_IMARIS_LOG", "").strip().upper()
    if not level_name:
        # Stay silent by default and avoid "no handlers could be found".
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        return logger

    level = getattr(logging, level_name, logging.DEBUG)
    logger.setLevel(level)
    logger.propagate = False

    log_file = os.environ.get("NAPARI_IMARIS_LOG_FILE", "").strip()
    if log_file:
        handler = logging.FileHandler(log_file, mode="w")
    else:
        handler = logging.StreamHandler(sys.stderr)

    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)s [%(threadName)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.info(
        "napari-imaris-loader verbose logging enabled (level=%s, file=%s)",
        level_name,
        log_file or "<stderr>",
    )
    return logger


def debug_enabled():
    """True when per-read instrumentation should be wired up."""
    return logger.isEnabledFor(logging.DEBUG)


@contextmanager
def timed_operation(operation_name):
    """Log START/END markers around a block and the elapsed wall-clock time.

    Mirrors the ``timed_operation`` helper in the related ``shifter`` project so
    GUI-driven actions (e.g. reloading at a new resolution level) can be timed
    to find which step hangs.  No-op cost when logging is disabled.
    """
    if not logger.isEnabledFor(logging.INFO):
        yield
        return
    logger.info("START %s", operation_name)
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.info(
            "END   %s (%.1f ms)", operation_name, (time.perf_counter() - start) * 1000
        )


def _fmt_slice(s):
    if isinstance(s, slice):
        step = "" if s.step in (None, 1) else ":{}".format(s.step)
        return "{}:{}{}".format(s.start, s.stop, step)
    return repr(s)


def _fmt_key(key):
    if isinstance(key, tuple):
        return "(" + ", ".join(_fmt_slice(k) for k in key) + ")"
    return _fmt_slice(key)


class TimedReader:
    """Transparent wrapper around an ``ims`` object that times every read.

    napari/dask call ``__getitem__`` once for each tile the canvas needs.
    Logging the duration and size of each call shows precisely how long the
    plugin spends fetching data versus how long is spent elsewhere (e.g. in
    napari's rendering).  Only used when DEBUG logging is enabled so it has no
    effect on normal runs.
    """

    def __init__(self, reader, level):
        # Use object.__setattr__ to avoid recursing through __getattr__.
        object.__setattr__(self, "_reader", reader)
        object.__setattr__(self, "_level", level)

    def __getattr__(self, name):
        # Delegate shape/dtype/ndim/chunks/etc. to the wrapped ims object.
        return getattr(self._reader, name)

    def __getitem__(self, key):
        start = time.perf_counter()
        result = self._reader[key]
        elapsed_ms = (time.perf_counter() - start) * 1000
        shape = getattr(result, "shape", None)
        nbytes = getattr(result, "nbytes", 0)
        logger.debug(
            "read level=%s key=%s -> shape=%s (%.2f MiB) in %.1f ms",
            self._level,
            _fmt_key(key),
            shape,
            nbytes / 2 ** 20,
            elapsed_ms,
        )
        return result
