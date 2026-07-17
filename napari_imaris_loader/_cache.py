# -*- coding: utf-8 -*-
"""A small, thread-safe chunk cache for the imaris reader.

Profiling showed the dominant interactive cost is re-reading the *same*
HDF5 chunks.  Two patterns repeat constantly while scrubbing a zoomed-in view:

* The native IMS chunks are 32 planes deep, so napari rendering a single 2D
  plane makes dask read the whole 32-deep chunk.  Every other plane in that
  band maps to the *identical* dask chunk key, yet nothing caches it - so
  scrolling within a band re-decompresses the same ~1 MiB chunk up to 32 times.
* Panning or zooming back to a region re-requests tiles that were just read.

A modest LRU keyed by ``(level, normalized_key)`` turns all of those repeats
into RAM hits.  It is opt-in via ``NAPARI_IMARIS_CACHE_MB`` (default on at a
safe size; set to ``0`` to disable) and shared across every resolution level
and channel so the memory budget is global.
"""

import logging
import threading
from collections import OrderedDict

from ._logging import logger


def _normalize_key(key):
    """Turn a ``__getitem__`` key into a hashable, comparable form.

    Slices are not hashable in a way that distinguishes start/stop/step
    cleanly for our purposes, so encode them explicitly.
    """
    if isinstance(key, tuple):
        return tuple(_normalize_key(k) for k in key)
    if isinstance(key, slice):
        return ('slice', key.start, key.stop, key.step)
    return key


class ReadCache:
    """Byte-bounded LRU cache of decompressed chunk arrays (thread-safe)."""

    _STATS_EVERY = 256  # log running stats once per this many requests

    def __init__(self, max_bytes):
        self.max_bytes = int(max_bytes)
        self._lock = threading.Lock()
        self._store = OrderedDict()
        self.cur_bytes = 0
        self.hits = 0
        self.misses = 0
        # >0 suppresses writes (e.g. during a bulk scrub materialisation that
        # would otherwise evict hot fine-level chunks); reads still hit.
        self._paused = 0

    def get(self, key):
        with self._lock:
            arr = self._store.get(key)
            if arr is not None:
                self._store.move_to_end(key)
                self.hits += 1
            else:
                self.misses += 1
            # Log running stats under the lock (accurate counters) and only
            # when DEBUG is actually enabled, so the default hot path pays
            # nothing.
            total = self.hits + self.misses
            if total % self._STATS_EVERY == 0 and logger.isEnabledFor(logging.DEBUG):
                rate = (self.hits / total * 100.0) if total else 0.0
                logger.debug(
                    "cache stats: hit_rate=%.1f%% (hits=%d misses=%d) "
                    "entries=%d %.0f MiB",
                    rate, self.hits, self.misses, len(self._store),
                    self.cur_bytes / 2 ** 20,
                )
            return arr

    def pause_writes(self):
        with self._lock:
            self._paused += 1

    def resume_writes(self):
        with self._lock:
            if self._paused > 0:
                self._paused -= 1

    def put(self, key, arr):
        nbytes = int(getattr(arr, 'nbytes', 0) or 0)
        # Skip empty probe reads and anything that cannot fit on its own.
        if nbytes <= 0 or nbytes > self.max_bytes:
            return
        with self._lock:
            if self._paused:
                return
            if key in self._store:
                self._store.move_to_end(key)
                return
            self._store[key] = arr
            self.cur_bytes += nbytes
            while self.cur_bytes > self.max_bytes and self._store:
                _, old = self._store.popitem(last=False)
                self.cur_bytes -= int(getattr(old, 'nbytes', 0) or 0)

    def stats(self):
        with self._lock:
            total = self.hits + self.misses
            rate = (self.hits / total * 100.0) if total else 0.0
            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': rate,
                'entries': len(self._store),
                'mib': self.cur_bytes / 2 ** 20,
            }


class CachingReader:
    """Transparent wrapper that serves repeated chunk reads from a ReadCache.

    Wraps an ``ims`` object (optionally already wrapped by ``TimedReader`` so
    actual reads are still timed).  On a hit it returns the stored array and
    logs nothing expensive; on a miss it delegates, stores the result, and
    periodically logs the running hit rate so a session log shows how much the
    cache helped.
    """

    def __init__(self, reader, level, cache):
        object.__setattr__(self, "_reader", reader)
        object.__setattr__(self, "_level", level)
        object.__setattr__(self, "_cache", cache)

    def __getattr__(self, name):
        return getattr(self._reader, name)

    def __getitem__(self, key):
        ckey = (self._level, _normalize_key(key))
        cached = self._cache.get(ckey)
        if cached is not None:
            return cached
        result = self._reader[key]
        self._cache.put(ckey, result)
        return result
