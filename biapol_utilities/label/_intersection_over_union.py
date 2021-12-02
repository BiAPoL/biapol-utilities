# -*- coding: utf-8 -*-

import numpy as np
from sklearn import metrics


def intersection_over_union_matrix(label_image_x, label_image_y):
    """Generate a matrix with intersection over union of all label pairs.

    How it works:
    The overlap matrix is a lookup table of the area of intersection
    between each set of labels (true and predicted). The true labels
    are taken to be along axis 0, and the predicted labels are taken
    to be along axis 1. The sum of the overlaps along axis 0 is thus
    an array giving the total overlap of the true labels with each of
    the predicted labels, and likewise the sum over axis 1 is the
    total overlap of the predicted labels with each of the true labels.
    Because the label 0 (background) is included, this sum is guaranteed
    to reconstruct the total area of each label. Adding this row and
    column vectors gives a 2D array with the areas of every label pair
    added together. This is equivalent to the union of the label areas
    except for the duplicated overlap area, so the overlap matrix is
    subtracted to find the union matrix.

    Source: [#]_

    Parameters
    ----------
    label_image_x: ND-array, int
        label image, where 0=background; 1,2... are label masks
    label_image_y: ND-array, int
        label image, where 0=background; 1,2... are label masks

    Returns
    -------
    iou: ND-array, float
        matrix of IOU pairs of size [x.max()+1, y.max()+1]

    References
    ----------
    .. [#] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    # Calculate overlap matrix
    overlap = metrics.confusion_matrix(label_image_x.ravel(),
                                       label_image_y.ravel())

    # Measure correctly labeled pixels
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)

    # Caluclate intersection over union
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0

    return iou
