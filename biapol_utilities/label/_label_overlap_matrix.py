# -*- coding: utf-8 -*-

import numpy as np
from sklearn import metrics

def label_overlap_matrix_numpy(label_image_x, label_image_y):
    """ 
    Legacy function to calculate label overlaps between masks in x and y.
    May be deprecated in future versions of biapol_utilities.
    
    From: https://github.com/MouseLand/cellpose/blob/6fddd4da98219195a2d71041fb0e47cc69a4b3a6/cellpose/metrics.py#L130
    
    Parameters
    ----------
    x: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    y: ND-array, int
        where 0=NO masks; 1,2... are mask labels

    Returns
    -------
    overlap: ND-array, int
        matrix of pixel overlaps of size [x.max()+1, y.max()+1]
    """

    label_image_x = label_image_x.ravel()
    label_image_y = label_image_y.ravel()
    
    overlap = np.zeros((1 + label_image_x.max(), 1 + label_image_y.max()), dtype=np.uint)
    
    for i in range(len(label_image_x)):
        overlap[label_image_x[i], label_image_y[i]] += 1
    
    return overlap

def label_overlap_matrix(label_image_x, label_image_y):
    
    """ Get pixel overlaps between masks in x and y
    
    From: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    
    Parameters
    ----------
    x: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    y: ND-array, int
        where 0=NO masks; 1,2... are mask labels

    Returns
    -------
    overlap: ND-array, int
        matrix of pixel overlaps of size [x.max()+1, y.max()+1]
    """
    
    
    # Flatten input arrays
    label_image_x = label_image_x.ravel()
    label_image_y = label_image_y.ravel()
    
    # Calculate confusion matrix
    overlap = metrics.confusion_matrix(label_image_x, label_image_y)
    
    # Remove empty rows/columns
    overlap = overlap[~np.all(overlap == 0, axis=1)]
    overlap = overlap[:, ~np.all(overlap == 0, axis=0)]
    
    return overlap
