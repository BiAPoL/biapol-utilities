# -*- coding: utf-8 -*-

import numpy as np
from ._intersection_over_union import intersection_over_union

def match_labels_stack(label_stack, method=intersection_over_union, **kwargs):
    """Match labels from subsequent slices with specified method

    Parameters
    ----------
    masks : 3D-array, int
        Stack of stacks to be stitched with axis order ZYX
    method : str, optional
        Method to be used for stitching the masks. The default is 'iou' with a stitch threshold of 0.25.
        *stitch_threshold* : float
            Threshold value for iou value above which two labels are considered identical.
            The default value is 0.25

    Returns
    -------
    masks : 3D-array, int
        Stack of stitched masks
    """

    if method == intersection_over_union:
        
        # iterate over masks
        for i in range(len(label_stack)-1):
            label_stack[i+1] = match_labels(label_stack[i], label_stack[i+1],
                                            method=method, **kwargs)
            
    return label_stack

def match_labels(label_image_x, label_image_y, method=intersection_over_union, **kwargs):
    """Match labels in label_image_y with labels in label_image_x based on similarity
    as defined by the passed method.
    
    Parameters
    ----------
    imageA : nd-array
        Image that should serve as a reference for label-matching
    imageB : nd-array
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
    
    if method == intersection_over_union:
        threshold = kwargs.get('iou_threshold', 0.25)
    
    
    # Calculate image similarity matrix img_sim based on chosen method
    img_sim = method(label_image_y, label_image_x)[1:,1:]
    mmax = label_image_x.max()
    
    if img_sim.size > 0:
        
        # Keep only ious above threshold
        img_sim[img_sim < threshold] = 0.0
        img_sim[img_sim < img_sim.max(axis=0)] = 0.0
        
        # Pick value with highest IoU value
        istitch = img_sim.argmax(axis=1) + 1
        ino = np.nonzero(img_sim.max(axis=1)==0.0)[0]  # Find unpaired labels
        
        # append unmatched labels and background to lookup table
        istitch[ino] = np.arange(mmax+1, mmax+len(ino)+1, 1, int)  
        mmax += len(ino)
        istitch = np.append(np.array(0), istitch)
        
        return istitch[label_image_y]
