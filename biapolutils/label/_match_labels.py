# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 11:58:34 2021

@author: Johannes Müller, Marcelo Zoccoler, Robert Haase

DFG funded cluster of excellence "Physics of Life", TU Dresden, Dresden, Germany
"""

import numpy as np
import tqdm
from ._intersection_over_union import intersection_over_union
from skimage import io
import os

def match_labels_stack(stack, method='iou', **kwargs):
    """
    Match labels from subsequent slices with specified method

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

    if method == 'iou':
        
        # iterate over masks
        for i in tqdm.tqdm(range(len(stack)-1), desc='Stitching slices'):
            stack[i+1] = match_labels(stack[i], stack[i+1], method=method, **kwargs)
            
    return stack

def match_labels(imageA, imageB, method='iou', **kwargs):
    """
    
    Match labels in two labelled images with each other
    
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
    
    if method == 'iou':
        iou_threshold = kwargs.get('iou_threshold', 0.25)
    
        iou = intersection_over_union(imageB, imageA)[1:,1:]
        mmax = imageA.max()
        
        if iou.size > 0:
            
            # Keep only ious above threshold
            iou[iou < iou_threshold] = 0.0
            iou[iou < iou.max(axis=0)] = 0.0
            
            # Pick value with highest IoU value
            istitch = iou.argmax(axis=1) + 1
            ino = np.nonzero(iou.max(axis=1)==0.0)[0]  # Find unpaired labels
            
            # append unmatched labels and background to lookup table
            istitch[ino] = np.arange(mmax+1, mmax+len(ino)+1, 1, int)  
            mmax += len(ino)
            istitch = np.append(np.array(0), istitch)
            
            return istitch[imageB]
    



    