# -*- coding: utf-8 -*-


import biapol_utilities as bputils
import numpy as np

def test_iou():
    
    size = 10
    a = np.arange(1, size, 1, dtype=np.uint8)
    b = np.arange(1, size, 1, dtype=np.uint8)
    np.random.shuffle(b)
    
    iou = bputils.label.intersection_over_union(a, b)
    
    assert(iou.shape == (size, size))
    
    

