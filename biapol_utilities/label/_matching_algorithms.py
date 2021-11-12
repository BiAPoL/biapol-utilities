# -*- coding: utf-8 -*-

import numpy as np


def max_similarity(label_image_x, label_image_y, similarity_matrix):
    """
    Matches labels in two input label images (label_image_x and label_image_y)
    based on the maximal value in the similarity_matrix.

    Parameters
    ----------
    label_image_x : ND-array, int
        labelled input image which serves as reference image
    label_image_y : ND-array, int
        Labbelled input image, the labels of which will be matched up with the
        labels in label_image_x based on the chosen input metric
    metric_matrix : 2D-array, float
        matrix containing the metric that defines the similarity of labels
        in label_image_x and label_image_y, with axis=0 referring to the labels
        of label_image_x. In other words, the entry similarity_matrix[i, j]
        corresponds to the "similarity" between label i in label_image_x and
        label j in label_image_y.

    Returns
    -------
    ND-array
        Relabelled version of input_image_y. Unmatched labels are appended to
        the set of matched labels and thus, the total number of labels may
        increase.

    """

    # Suppress non-maximal entries
    similarity_matrix[similarity_matrix < similarity_matrix.max(axis=0)] = 0.0

    mmax = label_image_x.max()

    if similarity_matrix.size > 0:
        # Pick value with highest IoU value
        istitch = similarity_matrix.argmax(axis=1) + 1

        # Find unpaired labels
        ino = np.nonzero(similarity_matrix.max(axis=1) == 0.0)[0]

        # append unmatched labels and background to lookup table
        istitch[ino] = np.arange(mmax+1, mmax+len(ino)+1, 1, int)
        mmax += len(ino)
        istitch = np.append(np.array(0), istitch)

        return istitch[label_image_y]
