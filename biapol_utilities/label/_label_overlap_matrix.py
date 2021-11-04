# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 11:55:46 2021

@author: Johannes Müller, Bio-image Analysis Technology Development Group at 
DFG Cluster of Excellence "Physics of Life", TU Dresden
"""

import numpy as np

def label_overlap_matrix(label_image_x, label_image_y):
    """ fast function to get pixel overlaps between masks in x and y 
    
    From: https://github.com/MouseLand/cellpose/blob/6fddd4da98219195a2d71041fb0e47cc69a4b3a6/cellpose/metrics.py#L130
    
    Parameters
    ------------
    x: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    y: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    Returns
    ------------
    overlap: ND-array, int
        matrix of pixel overlaps of size [x.max()+1, y.max()+1]
    
    """
    # put label arrays into standard form then flatten them 
#     x = (utils.format_labels(x)).ravel()
#     y = (utils.format_labels(y)).ravel()
    label_image_x = label_image_x.ravel()
    label_image_y = label_image_y.ravel()
    
    # preallocate a 'contact map' matrix
    overlap = np.zeros((1 + label_image_x.max(), 1 + label_image_y.max()), dtype=np.uint)
    
    # loop over the labels in x and add to the corresponding
    # overlap entry. If label A in x and label B in y share P
    # pixels, then the resulting overlap is P
    # len(x)=len(y), the number of pixels in the whole image 
    for i in range(len(label_image_x)):
        overlap[label_image_x[i], label_image_y[i]] += 1
    return overlap