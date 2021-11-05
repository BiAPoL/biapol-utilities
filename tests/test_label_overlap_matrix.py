# -*- coding: utf-8 -*-

import biapol_utilities as bputils
import numpy as np


def test_label_overlap_matrix():
    
    # create test arrays
    a = np.asarray([[1, 1, 0],
                    [1, 2, 0],
                    [0, 1, 0]])
    
    b = np.asarray([[1, 2, 0],
                    [1, 1, 0],
                    [0, 1, 0]])
    
    
    overlap = bputils.label.label_overlap_matrix(a, b)
    
    correct_result = np.asarray([[4, 0, 0],
                                 [0, 3, 1],
                                 [0, 1, 0]])
    
    match = [x == y for x, y in zip(overlap, correct_result)]
    
    assert(all(match))