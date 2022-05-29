"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/plugins/guides.html?#readers
"""
from functools import partial
from openslide import OpenSlide, open_slide, deepzoom
import numpy as np
from scipy import linalg
import dask.array as da
from dask import delayed

def getHEDfromRGB(rgb):
    # Conversion matrix for RGB-HED
    rgb_from_hed = np.array([[0.65, 0.70, 0.29],
                            [0.07, 0.99, 0.11],
                            [0.27, 0.57, 0.78]])
    hed_from_rgb = linalg.inv(rgb_from_hed)

    # Modifying _prepare_colorarray and [0,255] -> [0,1]
    rgb = da.asanyarray(rgb) / 255
    # log artifiacts
    da.maximum(rgb, 1E-6, out=rgb)
    # Compensate sum
    log_adjust = da.log(1E-6)
    hed = (da.log(rgb)/log_adjust) @ hed_from_rgb
    da.maximum(hed, 0, out=hed)
    
    return hed




def napari_get_reader(path, in_memory:bool):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # if we know we cannot read the file, we immediately return None.
    if not path.endswith(".svs"):
        return None

    # otherwise we return the *function* that can read ``path``.
    return partial(reader_function, in_memory=in_memory)


def reader_function(path, in_memory : bool):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of layer.
        Both "meta", and "layer_type" are optional. napari will default to
        layer_type=="image" if not provided
    """
    # Load the Img using AICS
    img = open_slide(path)
    gen = deepzoom.DeepZoomGenerator(img, tile_size = 1000, overlap = 0, limit_bounds=False)
    levels = len(img.level_dimensions)

    allow_unknown_chunksizes=False

    imgPy = []
    labPy = []
    for level in range(levels):
        max_level = gen.level_count - 1 - 2 * level
        n_tiles_x, n_tiles_y = gen.level_tiles[max_level]
        get_tile = lambda level,i,j: np.array(gen.get_tile(level,(i,j))).transpose((1,0,2))
        sample_tile = get_tile(max_level,0,0)
        sample_tile_shape = sample_tile.shape
        dask_get_tile = delayed(get_tile, pure=True)
        arr = (da.concatenate([da.concatenate([da.from_delayed(dask_get_tile(max_level,i,j),sample_tile_shape,np.uint8) for j in range(n_tiles_y)],allow_unknown_chunksizes=allow_unknown_chunksizes,axis=1) for i in range(n_tiles_x )],allow_unknown_chunksizes=allow_unknown_chunksizes))#.transpose([1,0,2]))

        imgPy.append(arr)

        if level == 0:
            rgb = arr.copy()
            hedarr = getHEDfromRGB(rgb)
            # Do the thresholding
            h_mask = hedarr[:,:,0]
            h_mask[h_mask >= 0.042] = 1

        else:
            # If not zoomed in enough, no mask
            h_mask = da.zeros(shape=arr[:,:,0].shape)
        
        labPy.append(h_mask.astype("uint8"))


    # For now, ignore metadata
    add_kwargs = {}

    layer_type = "image"  # optional, default is "image"
    return [(imgPy, add_kwargs, "image"), (labPy, {"name":"Nuclei"}, "labels")]
