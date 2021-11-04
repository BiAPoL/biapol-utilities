# -*- coding: utf-8 -*-

import numpy as np
import biabiapol_utilities as bputils

def test_match_labels():
    
    labels_x = np.asarray([1, 1, 1, 0, 0, 2, 2, 4, 4, 4, 0], dtype=np.uint8)
    labels_y = labels_x * 2
    
    labels_y_matched = bputils.label.match_labels(labels_x, labels_y)
    
    # calculate number of overlapping matches
    match = [x == y for x,y in zip(labels_x, labels_y_matched)]
    
    assert(all(match))

