# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:00:51 2021

@author: Johannes Müller, Marcelo Zoccoler, Robert Haase

DFG funded cluster of excellence "Physics of Life", TU Dresden, Dresden, Germany
"""


import numpy as np
import tqdm
from ._intersection_over_union import intersection_over_union

def stitch3Dto3D(masks, method='iou', **kwargs):
    """ stitch 3D masks into 4D volume with stitch_threshold on IOU 
    From: https://github.com/MouseLand/cellpose/blob/6fddd4da98219195a2d71041fb0e47cc69a4b3a6/cellpose/utils.py#L352
    """
    
    N_frames = masks.shape[0]
    
    if method == 'iou':
        stitch_threshold = kwargs.get('stitch_threshold', 0.25)
        
        mmax = masks[0].max()
        
        # Iterate over timeframes
        for i in tqdm.tqdm(range(N_frames - 1), desc='Stitching frames'):
            
            # calculate iou between masks
            iou = intersection_over_union(masks[i+1], masks[i])[1:,1:]  # ignore background label
            
            if iou.size > 0:
    
