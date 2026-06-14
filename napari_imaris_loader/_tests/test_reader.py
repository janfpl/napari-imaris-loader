import os
import numpy as np
from napari_imaris_loader import napari_get_reader


# tmp_path is a pytest fixture
def test_reader(tmp_path='brain_crop3.ims'):
    """An example of how you might test your plugin."""

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),tmp_path)
    # Test whether we get a callable reader function
    #reader = napari_get_reader('/path/to/a/fake.ims')
    reader = napari_get_reader(path)
    assert callable(reader)

    # make sure we're delivering the right format
    layer_data = reader(path)
    assert isinstance(layer_data, list)
    assert isinstance(layer_data[0], tuple) and len(layer_data[0]) == 2
    assert isinstance(layer_data[0][0], list) and isinstance(layer_data[0][1], dict)


def test_reader_chunks_compute(tmp_path='brain_crop3.ims'):
    """Computing the returned dask arrays must not raise.

    Regression test for the 3D multicolor import error
    ("too many indices ... Array chunk size or shape is unknown").
    The reader builds dask arrays from a full-ndim (t,c,z,y,x) ims view; if
    the per-chunk loader squeezes singleton dims, the declared shape/chunks no
    longer match what is returned and dask raises when napari computes a slice.
    """
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), tmp_path)
    layer_data = napari_get_reader(path)(path)

    data = layer_data[0][0]
    levels = data if isinstance(data, list) else [data]
    # Every resolution level must compute to its declared shape.
    for level in levels:
        assert np.asarray(level).shape == level.shape


def test_get_reader_pass():
    reader = napari_get_reader("fake.file")
    assert reader is None
