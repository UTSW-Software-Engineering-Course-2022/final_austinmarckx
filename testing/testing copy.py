import os
from openslide import OpenSlide, open_slide, deepzoom
import numpy as np
from scipy import linalg
import napari
from skimage.io import imread
from dask import delayed
import dask.array as da

def getHEDfromRGB():
    
    # Conversion matrix for RGB-HED
    
    # Haematoxylin-Eosin-DAB colorspace
    # From original Ruifrok's paper: A. C. Ruifrok and D. A. Johnston,
    # "Quantification of histochemical staining by color deconvolution,"
    # Analytical and quantitative cytology and histology / the International
    # Academy of Cytology [and] American Society of Cytology, vol. 23, no. 4,
    # pp. 291-9, Aug. 2001.
    rgb_from_hed = np.array([[0.65, 0.70, 0.29],
                            [0.07, 0.99, 0.11],
                            [0.27, 0.57, 0.78]])
    hed_from_rgb = linalg.inv(rgb_from_hed)

    return
# https://github.com/qupath/qupath/wiki/Supported-image-formats
# https://programtalk.com/vs4/python/jlevy44/PathFlowAI/deprecated_work/build/lib/pathflowai/utils.py/


# https://github.com/qupath/qupath/wiki/Supported-image-formats
# https://programtalk.com/vs4/python/jlevy44/PathFlowAI/deprecated_work/build/lib/pathflowai/utils.py/
def svs2dask_array(svs_file, tile_size=1000, overlap=0, remove_last=True, allow_unknown_chunksizes=False):
	"""Convert SVS, TIF or TIFF to dask array.

	Parameters
	----------
	svs_file:str
		Image file.
	tile_size:int
		Size of chunk to be read in.
	overlap:int
		Do not modify, overlap between neighboring tiles.
	remove_last:bool
		Remove last tile because it has a custom size.
	allow_unknown_chunksizes: bool
		Allow different chunk sizes, more flexible, but slowdown.

	Returns
	-------
	dask.array
		Dask Array.

	>>> arr=svs2dask_array(svs_file, tile_size=1000, overlap=0, remove_last=True, allow_unknown_chunksizes=False)
	>>> arr2=arr.compute()
	>>> arr3=to_pil(cv2.resize(arr2, dsize=(1440,700), interpolation=cv2.INTER_CUBIC))
	>>> arr3.save(test_image_name)"""
	img=open_slide(svs_file)
	gen=deepzoom.DeepZoomGenerator(img, tile_size=tile_size, overlap=overlap, limit_bounds=True)
	
	#for dim in range(len(gen.level_dimensions)-1):
	max_level = len(gen.level_dimensions)-1
	#max_level = dim #len(gen.level_dimensions)-1
	n_tiles_x, n_tiles_y = gen.level_tiles[max_level]
	get_tile = lambda i,j: np.array(gen.get_tile(max_level,(i,j))).transpose((1,0,2))
	sample_tile = get_tile(0,0)
	sample_tile_shape = sample_tile.shape
	dask_get_tile = delayed(get_tile, pure=True)
	arr=da.concatenate([da.concatenate([da.from_delayed(dask_get_tile(i,j),sample_tile_shape,np.uint8) for j in range(n_tiles_y - (0 if not remove_last else 1))],allow_unknown_chunksizes=allow_unknown_chunksizes,axis=1) for i in range(n_tiles_x - (0 if not remove_last else 1))],allow_unknown_chunksizes=allow_unknown_chunksizes)#.transpose([1,0,2])
	
	return arr

def svs2dask_array_pyramid(img, gen, level, tile_size=1000, overlap=0, remove_last=True, allow_unknown_chunksizes=False):
	"""Convert SVS, TIF or TIFF to dask array.

	Parameters
	----------
	svs_file:str
		Image file.
	tile_size:int
		Size of chunk to be read in.
	overlap:int
		Do not modify, overlap between neighboring tiles.
	remove_last:bool
		Remove last tile because it has a custom size.
	allow_unknown_chunksizes: bool
		Allow different chunk sizes, more flexible, but slowdown.

	Returns
	-------
	dask.array
		Dask Array.

	>>> arr=svs2dask_array(svs_file, tile_size=1000, overlap=0, remove_last=True, allow_unknown_chunksizes=False)
	>>> arr2=arr.compute()
	>>> arr3=to_pil(cv2.resize(arr2, dsize=(1440,700), interpolation=cv2.INTER_CUBIC))
	>>> arr3.save(test_image_name)"""
	
	max_level = gen.level_count - 1 - 2 * level
	n_tiles_x, n_tiles_y = gen.level_tiles[max_level]
	get_tile = lambda level,i,j: np.array(gen.get_tile(level,(i,j))).transpose((1,0,2))
	sample_tile = get_tile(max_level,0,0)
	sample_tile_shape = sample_tile.shape
	dask_get_tile = delayed(get_tile, pure=True)
	arr = (da.concatenate([da.concatenate([da.from_delayed(dask_get_tile(max_level,i,j),sample_tile_shape,np.uint8) for j in range(n_tiles_y)],allow_unknown_chunksizes=allow_unknown_chunksizes,axis=1) for i in range(n_tiles_x )],allow_unknown_chunksizes=allow_unknown_chunksizes))#.transpose([1,0,2]))

	return arr

sm_path = "D:/repos/swe22_final/sm.svs"
#img=open_slide(sm_path)
#gen=deepzoom.DeepZoomGenerator(img)
#lvls = len(open_slide(sm_path).level_dimensions)
#viewer = napari.Viewer() 
#viewer.add_image( [svs2dask_array_pyramid(img, gen, level) for level in range(lvls)], contrast_limits=[0,255])

viewer = napari.Viewer()
viewer
#viewer.add_image( svs2dask_array(sm_path), contrast_limits=[0,255])
