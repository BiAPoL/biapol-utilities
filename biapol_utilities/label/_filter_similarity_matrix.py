# -*- coding: utf-8 -*-


def suppressed_similarity(matrix, threshold=0.25):
    """
    Suppress entries in a similarity matrix below a defined threshold.

    This method sets all entries in the passed matrix below a defined threshold
    to zero, whereas all other entries remain untouched.

    Parameters
    ----------
    matrix : 2D-array, float
        Matrix containing the metric that defines the similarity of labels
        in two images. The entry in similarity_matrix[i, j] corresponds
        to the "similarity" between label i in in image X and label j in
        an image Y
    threshold : float, optional
        Values below *threshold* will be set to zero in the similarity matrix.
        The default is 0.25.

    Returns
    -------
    matrix : 2D-array, float
        Similarity matrix with suppressed values

    """
    if not len(matrix.shape) == 2:
        raise ValueError('Dimension of similarity matrix was '
                         f'expected NxD but was found {matrix.shape}')

    if threshold < 0 or threshold >= 1:
        raise ValueError('Provided threshold must be between' +
                         f'0 and 1 but was {threshold}')

    # Keep only IoUs above threshold
    matrix[matrix < threshold] = 0.0

    return matrix
