# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 16:49:37 2021

@author: alpha
"""

import napari, os
from magicgui import magic_factory
from napari_plugin_engine import napari_hook_implementation
# from .h5layer import layerH5
from .reader import ims_reader
from ._logging import configure_logging, logger, timed_operation
from .progressive_loading_widget import progressive_loading, purge_scrub_layers
from .detail_cap_widget import detail_cap, reset_detail_cap
import dask.array as da
from typing import List
from napari.layers import Image


def _detect_resolution_levels(viewer):
    """Return the number of resolution levels of the loaded IMS data.

    The reader stamps ``resolutionLevels`` into each layer's metadata.  We use
    that so the slider only offers levels that actually exist.  Returns ``None``
    when no IMS layer is present yet.
    """
    for layer in viewer.layers:
        levels = getattr(layer, 'metadata', {}).get('resolutionLevels')
        if levels:
            return int(levels)
    return None


def _resolution_widget_init(widget):
    """Keep the ``lowest_resolution_level`` slider bounded to the levels that
    actually exist in the currently loaded file (instead of a hard-coded 0-9).
    """

    def _refresh(*_):
        try:
            viewer = widget.viewer.value
            if viewer is None:
                return
            levels = _detect_resolution_levels(viewer)
            if not levels:
                return
            slider = widget.lowest_resolution_level
            slider.max = max(0, levels - 1)
            if slider.value > slider.max:
                slider.value = slider.max
        except Exception:
            # Never let dynamic UI tuning break the widget itself.
            pass

    def _hook_viewer(*_):
        viewer = widget.viewer.value
        if viewer is None:
            return
        # Avoid connecting more than once.
        if getattr(widget, '_imaris_levels_hooked', False):
            _refresh()
            return
        try:
            viewer.layers.events.inserted.connect(_refresh)
            viewer.layers.events.removed.connect(_refresh)
            widget._imaris_levels_hooked = True
        except Exception:
            pass
        _refresh()

    try:
        widget.viewer.changed.connect(_hook_viewer)
    except Exception:
        pass
    _hook_viewer()


@magic_factory(auto_call=False,call_button="update",
                widget_init=_resolution_widget_init,
                lowest_resolution_level={'min': 0,'max': 9,
                                  'tooltip':'''Important only for 3D rendering.
                                  Higher number is lower resolution.'''
                                  }
                )
def resolution_change(
    viewer: napari.Viewer,
    lowest_resolution_level: int
) -> 'napari.types.LayerDataTuple':
    
    ''' 
    This panel provides a tool for reloading the IMS data after selecting
    the lowest resolution level that will be included in the multiscale series.
    Higher numbers (ie higher on the pyramid) = lower resolution.  
    
    This is important for 3D rendering.  If one prefers higher resolution
    3D rendering, they can choose a lower number, update the viewer, then
    selecting 3D rendering.
    '''
    
    configure_logging()
    logger.info("resolution_change widget invoked: lowest_resolution_level=%s",
                lowest_resolution_level)

    ## Remove any scrub companion layers first.  They carry no 'fileName'
    ## metadata and overlay the real data, so leaving them in place both
    ## crashes the source-layer lookup below and leaves stale low-res copies
    ## over the freshly reloaded data.
    purge_scrub_layers(viewer)

    ## Forget any detail-cap originals; the reloaded pyramid replaces them.
    reset_detail_cap(viewer)

    ## Locate a real IMS layer to reload from (the reader stamps 'fileName'
    ## into each layer's metadata; companion/other layers won't have it).
    source_path = None
    for layer in viewer.layers:
        source_path = getattr(layer, 'metadata', {}).get('fileName')
        if source_path:
            break
    if not source_path:
        logger.warning("resolution_change: no IMS layer with a 'fileName' found; "
                       "nothing to reload")
        return

    ## Load data for IMS file using the loader function
    try:
        with timed_operation("ims_reader reload (resLevel=%s)" % lowest_resolution_level):
            tupleOut = ims_reader(
                source_path,
                colorsIndependant=True,
                resLevel=lowest_resolution_level
                )
    except ValueError as e:
        logger.warning("resolution_change reload failed: %s", e)
        print(e)
        return
    '''tupleOut is a tuple for each channel in the ims file
    structured as: [ ( [listOfMultiscaleDataCh1],metaDataDict ), 
                   ( [listOfMultiscaleDataCh2],metaDataDict ) ]
    '''
    # print(tupleOut)
    
    ## Determine Channel Names extracted from IMS file
    channelNames = []
    for tt in tupleOut:
        channelNames.append(tt[1]['name'])
    # print(channelNames)
    
    # for idx in viewer.layers:
    #     print(viewer.layers[str(idx)].data)
    
    ## Collect viewer state info about each layer with the same names extracted 
    ## from the ims file.  Add these parameters to the metadata extracted from file.
    ## Then delete the old layers
    
    ## Force viewer into 2D mode to avoid interpolation and
    ## axes don't match errors.  Not sure why these are caused
    if viewer.dims.ndisplay == 3:
        viewer.dims.ndisplay = 2
        
    for num,idx in enumerate(channelNames):

        layer = viewer.layers[str(idx)]
        tmp = {
            'opacity':layer.opacity,
            'gamma':layer.gamma,
            'colormap':layer.colormap,
            'blending':layer.blending,
            'visible':layer.visible,
            'rendering':layer.rendering

            # 'contrast_limits_range':layer.contrast_limits

            }

        # napari 0.5+ split the single ``interpolation`` attribute into
        # ``interpolation2d`` and ``interpolation3d``.  Preserve whichever the
        # installed napari exposes so the rebuilt layers keep the user's choice
        # and we stay compatible with older versions.
        if hasattr(layer, 'interpolation2d'):
            tmp['interpolation2d'] = layer.interpolation2d
            tmp['interpolation3d'] = layer.interpolation3d
        else:
            tmp['interpolation'] = layer.interpolation

        tupleOut[num][1].update(tmp)

        del(viewer.layers[str(idx)])

    ## Return the tuple data that will be loaded into the viewer
    return tupleOut

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [resolution_change, progressive_loading, detail_cap]

