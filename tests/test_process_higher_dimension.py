# -*- coding: utf-8 -*-

import numpy as np
from biapol_utilities.utilities import process_higher_dimension
from skimage.transform import warp_polar

def test_process_higher_dimension():
    # 4D image
    image = np.arange(120).reshape((2,3,4,5))

    #  Build output with appropriate shape
    new_shape = [image.shape[0], image.shape[1], 360, min(image.shape[2], image.shape[3])]
    image_polar = np.zeros(new_shape)
    for t in range(image.shape[0]):
        for z in range(image.shape[1]):
            image_polar[t, z] = warp_polar(image[t, z])

    image_polar2 = process_higher_dimension(image, warp_polar)

    assert np.array_equal(image_polar, np.asarray(image_polar2))