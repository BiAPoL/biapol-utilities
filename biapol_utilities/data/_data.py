# -*- coding: utf-8 -*-

from skimage import io
import os

data_dir = os.path.abspath(os.path.dirname(__file__))

def blobs():
    
    """Gray-level "blobs" image [1].
    
    Can be used for segmentation and denoising examples.
    
    Returns
    -------
    blobs : (256, 254) uint8 ndarray
        Blobs image.

    References
    ----------
    .. [1] https://imagej.nih.gov/ij/images/
    """
    
    return io.imread(os.path.join(data_dir, "blobs.png"))
