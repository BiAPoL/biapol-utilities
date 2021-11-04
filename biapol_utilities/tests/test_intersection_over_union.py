# -*- coding: utf-8 -*-


import biapol_utilities as bputils
import numpy as np

def test_iou():
    
    size = 10
    a = np.random.randint(1, 10, size=size)
    b = np.random.randint(1, 10, size=size)
    
    iou = bputils.label.intersection_over_union(a, b)
    
    assert(iou.shape == (size, size))
    
    

