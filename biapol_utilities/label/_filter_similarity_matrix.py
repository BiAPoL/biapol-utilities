# -*- coding: utf-8 -*-
import numpy as np

def suppressed_maximal(matrix, threshold = 0.25):
    """
    Suppresses entries in a similarity matrix below a defined threshold and removes
    all non-maximal entries

    Parameters
    ----------
    matrix : 2D-array, float
        Matrix containing the metric that defines the similarity of labels
        in two images. The entry in similarity_matrix[i, j] corresponds
        to the "similarity" between label i in in image X and label j in
        an image Y
    threshold : float, optional
        Values below *threshold* will be set to zero in the similarity matrix. The default is 0.25.

    Returns
    -------
    matrix : 2D-array, float
        Similarity matrix with suppressed values

    """
    
    if not len(matrix.shape) == 2:
        raise ValueError(f'Dimension of similarity matrix was expected NxD but was found {matrix.shape}')
        
    if threshold < 0 or threshold >= 1:
        raise ValueError(f'Provided threshold must be between 0 and 1 but was {threshold}')
        # Keep only ious above threshold
    
    matrix[matrix < threshold] = 0.0
    matrix[matrix < matrix.max(axis=0)] = 0.0
    matrix[matrix < threshold] = 0
    
    return matrix