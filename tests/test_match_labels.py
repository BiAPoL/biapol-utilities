# -*- coding: utf-8 -*-

import numpy as np
import biapol_utilities as biau

def test_match_labels2():
    
    labels_x = np.asarray([1, 1, 1, 0, 0, 2, 2, 4, 4, 4, 0], dtype=np.uint8)
    labels_y = np.asarray([1, 1, 2, 0, 0, 3, 3, 4, 4, 4, 5], dtype=np.uint8)

    reference_y_matched = np.asarray([1, 1, 1, 0, 0, 2, 2, 4, 4, 4, 5], dtype=np.uint8)

    labels_y_matched = biau.label.match_labels(labels_x, labels_y)

    assert np.array_equal(reference_y_matched, labels_y_matched)
