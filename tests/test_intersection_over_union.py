# -*- coding: utf-8 -*-


import biapol_utilities as biau
import numpy as np

def test_iou():
    
    size = 10
    a = np.arange(1, size, 1, dtype=np.uint8)
    b = np.arange(1, size, 1, dtype=np.uint8)
    np.random.shuffle(b)
    
    iou = biau.label.intersection_over_union(a, b)
    
    assert(iou.shape == (len(a), len(b)))

def test_intersection_over_union_matrix():
    a = np.asarray([1, 1, 2, 2, 0, 0])
    b = np.asarray([3, 2, 2, 0, 0, 1])

    reference = np.asarray([[0.33, 0.5,  0.,   0.  ],
                            [0.,   0.,   0.33, 0.5 ],
                            [0.33, 0.,   0.33, 0.  ]])

    iou = biau.intersection_over_union(a, b)

    assert np.allclose(iou, reference, 0.02)
    

if __name__ == "__main__":
    test_iou()