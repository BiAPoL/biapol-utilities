# -*- coding: utf-8 -*-

import numpy as np
from biapol_utilities import measure

def test_label_wise_jaccard():
    
    a = np.asarray([5, 0, 0, 1, 1, 1, 2, 2])
    b = np.asarray([5, 0, 0, 1, 1, 1, 2, 3])
    
    JC = measure.labelwise_jaccard_score(a, b)
    
    assert(JC is not None)