# -*- coding: utf-8 -*-

import numpy as np
from skimage.segmentation import relabel_sequential
from ._intersection_over_union import intersection_over_union_matrix
from ._matching_algorithms import max_similarity, gale_shapley
from ._filter_similarity_matrix import suppressed_similarity
import tqdm

def match_labels_stack(label_stack, method=intersection_over_union_matrix, **kwargs):
    """Match labels from subsequent slices with specified method

    Parameters
    ----------
    label_stack : 3D-array, int
        Stack of 2D label images to be stitched with axis order ZYX
    method : str, optional
        Method to be used for stitching the masks. The default is intersection_over_union_matrix with a stitch threshold of 0.25.
        *stitch_threshold* : float
            Threshold value for iou value above which two labels are considered identical.
            The default value is 0.25

    Returns
    -------
    3D-array, int
        Stack of stitched masks
    """

    # iterate over stack of label images
    for i in tqdm.tqdm(range(len(label_stack)-1)):
        label_stack[i+1] = match_labels(label_stack[i], label_stack[i+1],
                                        metric_method=metric_method,
                                        filter_method=filter_method,
                                        matching_method=matching_method)

    return label_stack

def match_labels(label_image_x, label_image_y, method=intersection_over_union_matrix, **kwargs):
    """Match labels in label_image_y with labels in label_image_x based on similarity
    as defined by the passed method.
    
    Parameters
    ----------
    label_image_x : nd-array
        Image that should serve as a reference for label-matching
    label_image_y : nd-array
        Image the labels of which should be paired with labels from imageA
    method : str, optional
        Pairing method to be used. The default is 'iou' (intersection over union).
        
        *iou (intersection over union)*: The intersection over union is used to 
            measure the overlap between labels in both images. The label with
            highest overlap is chosen as counterpart label
            
            iou_streshold: float, optional
                Threshold value above which two labels are accepted as overlapping.
                The default value is 0.25

    Returns
    -------
    nd-array
        Processed version of imageB with labels corresponding to imageA.
    """
    
    if method == intersection_over_union_matrix:
        threshold = kwargs.get('iou_threshold', 0.25)
    
    # relabel label_image_y to keep overlap matrix small
    label_image_y, _, _ = relabel_sequential(label_image_y)

    # Calculate image similarity metric
    similarity_matrix = metric_method(label_image_y.ravel(),
                                      label_image_x.ravel())

    if similarity_matrix.shape[0] == 1098:
        print('Halt')

    # Force-match background with background
    similarity_matrix[0, :] = 0
    similarity_matrix[:, 0] = 0
    similarity_matrix[0, 0] = 1.0

    # Filter similarity metric matrix
    if filter_method is None:
        pass
    else:
        similarity_matrix = filter_method(similarity_matrix)

    # Apply matching technique
    output = matching_method(label_image_x, label_image_y, similarity_matrix)

    return output
