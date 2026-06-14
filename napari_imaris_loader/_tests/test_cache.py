"""Tests for the chunk read cache (_cache.py)."""

from napari_imaris_loader._cache import ReadCache, CachingReader, _normalize_key


class _FakeArr:
    """Minimal stand-in for a numpy/dask chunk with an ``nbytes`` attribute."""

    def __init__(self, nbytes, tag=None):
        self.nbytes = nbytes
        self.tag = tag


class _FakeReader:
    def __init__(self):
        self.calls = 0

    @property
    def shape(self):
        return (1, 2, 3)

    def __getitem__(self, key):
        self.calls += 1
        return _FakeArr(100, tag=key)


def test_normalize_key_encodes_slices():
    key = (slice(0, 32), 5, slice(128, 256, 2))
    assert _normalize_key(key) == (
        ('slice', 0, 32, None), 5, ('slice', 128, 256, 2),
    )


def test_cache_miss_then_hit():
    cache = ReadCache(10_000)
    assert cache.get(('k',)) is None  # miss
    arr = _FakeArr(100)
    cache.put(('k',), arr)
    assert cache.get(('k',)) is arr   # hit, same object
    assert cache.hits == 1
    assert cache.misses == 1


def test_cache_evicts_lru_when_over_budget():
    cache = ReadCache(250)  # fits two 100-byte entries, not three
    cache.put(('a',), _FakeArr(100))
    cache.put(('b',), _FakeArr(100))
    cache.put(('c',), _FakeArr(100))  # pushes total over budget -> evict 'a'
    assert cache.get(('a',)) is None
    assert cache.get(('b',)) is not None
    assert cache.get(('c',)) is not None


def test_cache_touch_keeps_recent_alive():
    cache = ReadCache(250)
    cache.put(('a',), _FakeArr(100))
    cache.put(('b',), _FakeArr(100))
    cache.get(('a',))                 # 'a' is now most-recently used
    cache.put(('c',), _FakeArr(100))  # evict LRU -> 'b', not 'a'
    assert cache.get(('a',)) is not None
    assert cache.get(('b',)) is None


def test_cache_skips_oversized_and_empty():
    cache = ReadCache(50)
    cache.put(('big',), _FakeArr(100))  # larger than whole budget -> skip
    cache.put(('zero',), _FakeArr(0))   # empty probe read -> skip
    assert cache.get(('big',)) is None
    assert cache.get(('zero',)) is None
    assert cache.cur_bytes == 0


def test_caching_reader_serves_repeats_and_delegates():
    cache = ReadCache(10_000)
    backing = _FakeReader()
    reader = CachingReader(backing, level=0, cache=cache)

    a = reader[(0, slice(0, 32), 5)]
    b = reader[(0, slice(0, 32), 5)]
    assert a is b                 # second read served from cache
    assert backing.calls == 1     # backing reader hit only once
    assert cache.hits == 1
    assert cache.misses == 1

    # Different key -> different backing read.
    reader[(1, slice(0, 32), 5)]
    assert backing.calls == 2

    # Attribute access is delegated to the wrapped reader.
    assert reader.shape == (1, 2, 3)


def test_caching_reader_separates_levels():
    cache = ReadCache(10_000)
    r0 = CachingReader(_FakeReader(), level=0, cache=cache)
    r1 = CachingReader(_FakeReader(), level=1, cache=cache)
    r0[(0, 0)]
    # Same getitem key but different level must not collide.
    assert cache.get((1, _normalize_key((0, 0)))) is None
    r1[(0, 0)]
    assert cache.get((0, _normalize_key((0, 0)))) is not None
    assert cache.get((1, _normalize_key((0, 0)))) is not None
