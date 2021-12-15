# -*- coding: utf-8 -*-

import biapol_utilities as biau
import numpy as np


def test_suppression():

    a = np.random.rand(100).reshape(10, -1)
    threshold = 0.5

    a_sup = biau.label.suppressed_similarity(a, threshold=threshold)

    assert(all(a_sup[a < threshold].ravel() == 0))


if __name__ == "__main__":
    test_suppression()
