# -*- coding: utf-8 -*-
import numpy as np

def suppression_threshold(matrix, threshold = 0.25):
    """
    Suppresses entries in a similarity matrix below a defined threshold

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
    
    matrix[matrix < threshold] = 0
    
    return matrix

def label_wise_maximum(matrix, axis=0):
    """
    Removes all non-maximal entries in the similarity matrix in a label-wise fashion.
    This means that the maximal value along a specified axis of the similarity matrix is detected,
    and all other entries are set to zero.

    Parameters
    ----------
    matrix : 2D-array
        Matrix containing the metric that defines the similarity of labels
        in two images. The entry in similarity_matrix[i, j] corresponds
        to the "similarity" between label i in in image X and label j in
        an image Y
    axis : int, optional
        Axis along which maxima will be detected. The default is 0.

    Returns
    -------
    matrix : 2D-array
        Matrix containing the metric that defines the maximal possible similarity of labels
        in two images.

    """
    
    if not len(matrix.shape) == 2:
        raise ValueError(f'Dimension of similarity matrix was expected NxD but was found {matrix.shape}')
    
    matrix[:, np.argmax(matrix, axis=axis)] = 0
    matrix = 1.0 * (matrix > 0)
    
    return matrix