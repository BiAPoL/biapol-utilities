# -*- coding: utf-8 -*-

import numpy as np
from sklearn import metrics

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

    label_image_x = label_image_x.ravel()
    label_image_y = label_image_y.ravel()
    
    overlap = metrics.confusion_matrix(label_image_x, label_image_y)
    
    return overlap
