# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 13:23:56 2021

@author: johan
"""

import os
import biapolutils
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
    path = r'D:\Documents\Promotion\Projects\2021_napari_quantify_segmentation\data\set1_unmatched'
    imageA = io.imread(os.path.join(path, 'ground_truth.tif'))
    imageB = io.imread(os.path.join(path, 'segmented.tif'))
    
    # image = np.zeros((2, imageA.shape[0], imageA.shape[1]), dtype=np.uint16)
    # image[0] = imageA
    # image[1] = imageB
    
    # output = biapolutils.label.match_labels_stack(image)
    output = biapolutils.label.match_labels(imageA, imageB)
    
    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    axes[0].imshow(imageA[35])
    axes[1].imshow(output[35])