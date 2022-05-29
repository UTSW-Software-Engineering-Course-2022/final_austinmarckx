from napari_plugin_engine import napari_hook_implementation
import numpy as np
from scipy import linalg
from ._reader import napari_get_reader
from ._svsOutMemoryHED import svsOutMemoryHED

@napari_hook_implementation
def rgb2hed():
    return getHEDfromRGB


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