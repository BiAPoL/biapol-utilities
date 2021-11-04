# -*- coding: utf-8 -*-


import numpy as np
from biapol_utilities import measure
import pandas as pd


def compare_labels(label_image_true, label_image_pred):
    """
    Compares two label images regarding metrics from sklearn

    Parameters
    ----------
    label_image_true : ndarray, int
        Ground truth label image of arbitrary dimensions.
    label_image_y : ndarray, int
        Predicted label image which will be compared to ground truth label image.

    Returns
    -------
    pandas dataframe with columns [label, jaccard-score]. Each row corresponds
    to the measured jaccard-score of the respective label.
    """
    
    # Flatten input
    label_image_true = label_image_true.flatten()
    label_image_pred = label_image_pred.flatten()
    
    if not label_image_true.shape == label_image_pred.shape:
        raise Exception(f'Input images must have same dimension but have {label_image_true.shape} and {label_image_pred.shape}')
    
    # get list of present labels in bothn images and allocate results
    labels = np.unique(np.hstack([label_image_true, label_image_pred]))
    df = pd.DataFrame(columns = ['label', 'jaccard_score'])
    
    # calculate Jaccard score
    jc_score = measure.labelwise_jaccard_score(label_image_true,
                                               label_image_pred)
    
    df['label'] = labels
    df['jaccard_score'] = jc_score
    
    return df