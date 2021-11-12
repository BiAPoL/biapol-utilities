# -*- coding: utf-8 -*-

from biapol_utilities import label
import numpy as np


def test_compare_labels():

    a = np.asarray([5, 0, 0, 1, 1, 1, 2, 2])
    b = np.asarray([5, 0, 0, 1, 1, 1, 2, 3])

    result = label.compare_labels(a, b)

    assert('jaccard_score' in result.columns)
    assert('dice_score' in result.columns)


def test_compare_labels2():

    a = np.asarray([5, 0, 0, 1, 1, 1, 2, 2])
    b = np.asarray([6, 0, 0, 1, 1, 1, 2, 3])

    result = label.compare_labels(a, b)

    assert(np.max(result.label) == np.max([a, b]))


def test_compare_labels3():

    a = np.asarray([5, 0, 0, 1, 1, 1, 2, 2])
    b = np.asarray([6, 0, 0, 1, 1, 1, 2, 3])

    result = label.compare_labels(a, b)

    assert(result[result.label == 0].jaccard_score.to_numpy()[0] == 1.0)


if __name__ == "__main__":
    test_compare_labels()
    test_compare_labels2()
    test_compare_labels3()
