# -*- coding: utf-8 -*-

from sklearn.metrics import jaccard_score
import numpy as np

def labelwise_jaccard_score(label_image_true, label_image_pred):
    
    """
    Calculates the label-wise jaccard score of two label images

    Parameters
    ----------
    label_image_true : nd-array, int
        Ground truth label image
    label_image_pred : nd-array, int
        predicted label image

    Returns
    -------
    jc_score : array, float
        array with i-th entry corresponding to jaccard-index of i-th label entry 
        in set of labels

    """
    np.issubdtype(np.uint32, np.integer)

    
    if not np.issubdtype(label_image_pred.dtype, np.integer):
        raise TypeError(f'Predicted label image must be int but is {label_image_pred.dtype}')
        
    if not np.issubdtype(label_image_true.dtype, np.integer):
        raise TypeError(f'Ground truth label image must be int but is {label_image_pred.dtype}')
    
    
    # Flatten input
    label_image_true = label_image_true.flatten()
    label_image_pred = label_image_pred.flatten()
    
    jc_score = jaccard_score(label_image_true, label_image_pred, average=None)
    
    return jc_score