# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 16:56:01 2021

@author: johan
"""


import biapol_utilities as bputils
import numpy as np


def test_match_labels_stack():
    
    a = np.asarray([[1, 1, 0],
                    [1, 1, 2],
                    [0, 0, 2]])
    b = a
    c = a * 2
    d = a * 3
    
    stack = [a, b, c, d]
    stack = np.stack(stack)
    
    # Check that labels in last slice of axis = 0 are identical
    matched_stack = bputils.label.match_labels_stack(stack)
    match = [x==y for x, y in zip(a.flatten(), matched_stack[-1].flatten())]

    assert(all(match))