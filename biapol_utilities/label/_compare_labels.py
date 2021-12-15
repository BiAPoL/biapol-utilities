# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import jaccard_score
import pandas as pd


def compare_labels(label_image_x, label_image_y):
    r"""
    Evaluate differences between two label images.

    Compares two label images to determine the
    label-wise Jaccard- and Dice scores.
    The Jaccard-score is defined as the intersection over union of two labelled
    images [#]_. The Dice score S can be derived from the Jaccard-score J
    through the following relation:

    .. math:: S = \frac{2J}{1+J}

    Parameters
    ----------
    label_image_x : ND-array, int
        label image of arbitrary dimensions.
    label_image_y : ND-array, int
        label image which will be compared to `label_image_x`, must have the
        same dimensions.

    Returns
    -------
    Pandas dataframe
        The function returns a pandas dataframe with columns `['label', 'jaccard_score', 'dice_score']`. Each row corresponds to the measured quantitiy (Jaccard- or Dice score) of the respective label.

    References
    ----------
    .. [#] https://en.wikipedia.org/wiki/Jaccard_index
    """
    # Convert image to 1D array. Makes elemt-wise comparison of arrays simpler.
    label_image_x = label_image_x.ravel()
    label_image_y = label_image_y.ravel()

    if not label_image_x.shape == label_image_y.shape:
        raise Exception('Input images must have same dimension but have '
                        f'{label_image_x.shape} and {label_image_y.shape}')

    # get list of present labels in bothn images and allocate results
    labels = np.unique(np.hstack([label_image_x, label_image_y]))
    df = pd.DataFrame(columns=['label', 'jaccard_score', 'dice_score'])

    # calculate Jaccard score
    jc_score = jaccard_score(label_image_x, label_image_y, average=None)
    dc_score = (2 * jc_score) / (1 + jc_score)

    df['label'] = labels
    df['jaccard_score'] = jc_score
    df['dice_score'] = dc_score

    return df
