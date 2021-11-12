# -*- coding: utf-8 -*-
import biapol_utilities as biau
import numpy as np

def test_iou():

    # Create two arrays of random length
    size = np.random.randint(5, 10)
    low = 1
    a = np.arange(low, size, 1, dtype=np.uint8)
    b = np.arange(low, size, 1, dtype=np.uint8)

    # Shuffle array
    np.random.shuffle(b)

    # delete random entries from a and b
    idx = np.random.randint(low, size-1, 1)
    b[idx] = 0
    a[idx] = 0

    iou = biau.label.intersection_over_union_matrix(a, b)

    assert(iou.shape == (len(np.unique([a, b])), len(np.unique([a, b]))))

def test_intersection_over_union_matrix():
    a = np.asarray([1, 1, 2, 2, 0, 0])
    b = np.asarray([3, 2, 2, 0, 0, 1])

    reference = np.asarray([[0.33, 0.5,  0.,   0.  ],
                            [0.,   0.,   0.33, 0.5 ],
                            [0.33, 0.,   0.33, 0.  ],
                            [0.,   0.,   0.,   0.  ]])

    iou = biau.intersection_over_union_matrix(a, b)

    assert np.allclose(iou, reference, 0.02)

if __name__ == "__main__":
    test_iou()
