# -*- coding: utf-8 -*-

import numpy as np
from sklearn import metrics

def label_overlap_matrix(label_image_x, label_image_y):
    
    """ Get pixel overlaps between masks in x and y
    
    From: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    
    Parameters
    ----------
    label_image_x: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    label_image_y: ND-array, int
        where 0=NO masks; 1,2... are mask labels

    Returns
    -------
    overlap: ND-array, int
        matrix of pixel overlaps of size [max(x, y) + 1, max(x, y) + 1], with 
        entry result[i, j] corresponding to the overlap of label i in image x 
        with label j in image y
    """
    
    
    # Make input arrays 1D
    label_image_x = label_image_x.ravel()
    label_image_y = label_image_y.ravel()
    
    # Calculate confusion matrix
    overlap = metrics.confusion_matrix(label_image_x, label_image_y)
    
    return overlap
