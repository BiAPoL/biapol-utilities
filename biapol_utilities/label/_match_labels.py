# -*- coding: utf-8 -*-

from skimage.segmentation import relabel_sequential
from ._intersection_over_union import intersection_over_union_matrix
from ._matching_algorithms import max_similarity
from ._filter_similarity_matrix import suppressed_similarity


def match_labels_stack(label_stack, method=intersection_over_union_matrix,
                       **kwargs):
    """
    Match labels from subsequent slices with specified method

    Parameters
    ----------
    label_stack : 3D-array, int
        Stack of 2D label images to be stitched with axis order ZYX
    method : callable, optional
        Method to be used for matching the labels. This function is supposed
        to return a binary matrix with `shape=(n+1, m+1)` corresponding to
        `match=1`, `no_match=0` of the `n` labels in a given slice and
        `m` labels in the following slice. The default is
        thresholded_intersection_over_union_matrix.

    Returns
    -------
    3D-array, int
        Stack of stitched labels
    """

    # iterate over masks
    for i in range(len(label_stack)-1):
        label_stack[i+1] = match_labels(label_stack[i], label_stack[i+1],
                                        method=method, **kwargs)

    return label_stack


def match_labels(label_image_x, label_image_y, **kwargs):
    """
    Match labels in label_image_y with labels in label_image_x based on
    similarity as defined by the passed method.

    Parameters
    ----------
    label_image_x : nd-array
        Image that should serve as a reference for label-matching
    label_image_y : nd-array
        Image the labels of which should be paired with labels from imageA
    method : callable, optional
        Method to be used for matching the labels. This function is supposed
        to return a binary matrix with `shape=(n+1, m+1)` corresponding to
        `match=1`, `no_match=0` of the `n` labels in a given slice and
        `m` labels in the following slice. The default is
        thresholded_intersection_over_union_matrix.

    Returns
    -------
    nd-array
        Processed version of label_image_y with labels corresponding to
        label_image_x.
    """

    method_metric = kwargs.get('metric', intersection_over_union_matrix)
    method_filter = kwargs.get('filter', suppressed_similarity)
    method_matching = kwargs.get('matching', max_similarity)

    # relabel label_image_y to keep overlap matrix small
    label_image_y, _, _ = relabel_sequential(label_image_y)

    # Calculate image similarity metric
    similarity_matrix = method_metric(label_image_y.flatten(),
                                      label_image_x.flatten())[1:, 1:]

    # Filter similarity metric matrix
    similarity_matrix = method_filter(similarity_matrix)

    # Apply matching technique
    output = method_matching(label_image_x, label_image_y, similarity_matrix)

    return output
