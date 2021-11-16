# -*- coding: utf-8 -*-

import numpy as np
from biapol_utilities import label


def test_match_labels():

    labels_x = np.asarray([1, 1, 1, 0, 0, 2, 2, 4, 4, 4, 0], dtype=np.uint8)
    labels_y = labels_x * 2

    matching_method1 = label.match_max_similarity
    matching_method2 = label.match_gale_shapley

    labels_y_matched1 = label.match_labels(labels_x, labels_y,
                                           matching_method=matching_method1)
    labels_y_matched2 = label.match_labels(labels_x, labels_y,
                                           matching_method=matching_method2)

    assert np.array_equal(labels_y_matched1, labels_x)
    assert np.array_equal(labels_y_matched2, labels_x)


def test_match_labels_2():
    labels_x = np.asarray([1, 1, 6, 0, 0, 3, 3, 4, 4, 4, 0], dtype=np.uint8)
    labels_y = np.asarray([1, 1, 3, 0, 0, 2, 2, 5, 5, 5, 4], dtype=np.uint8)

    reference_y_matched = np.asarray([1, 1, 6, 0, 0, 3, 3, 4, 4, 4, 7],
                                     dtype=np.uint8)

    matching_method = label.match_max_similarity
    labels_y_matched = label.match_labels(labels_x, labels_y,
                                          matching_method=matching_method)

    assert np.array_equal(labels_y_matched, reference_y_matched)


def test_match_labels_3():
    labels_x = np.asarray([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 0], dtype=np.uint8)
    labels_y = np.asarray([1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], dtype=np.uint8)

    reference_y_matched = np.asarray([1, 1, 4, 4, 4, 2, 2, 2, 3, 3, 3],
                                     dtype=np.uint8)

    matching_method = label.match_gale_shapley
    labels_y_matched = label.match_labels(labels_x, labels_y,
                                          matching_method=matching_method,
                                          filter_method=None)

    assert np.array_equal(labels_y_matched, reference_y_matched)


if __name__ == "__main__":
    test_match_labels_3()
    test_match_labels_2()
    test_match_labels()
