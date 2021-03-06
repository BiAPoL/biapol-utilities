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


def test_gale_shapley():

    labels_x = np.asarray([0, 1, 2, 3])
    labels_ref = np.asarray([0, 2, 1, 3])

    similarity_matrix = [[0.1, 0.0, 0.0, 0.0],
                         [0.0, 0.2, 0.5, 0.8],
                         [0.0, 0.9, 0.2, 0.5],
                         [0.0, 0.8, 0.7, 0.81]]

    similarity_matrix = np.asarray(similarity_matrix)

    output = label.match_gale_shapley(labels_x, labels_x, similarity_matrix)

    assert np.array_equal(output, labels_ref)

def test_gale_shapley2():

    labels_x = np.asarray([0, 1, 2, 3])
    labels_ref = np.asarray([0, 2, 3, 1])

    similarity_matrix = [[0.1, 0.0, 0.0, 0.0],
                         [0.0, 0.5, 0.51, 0.4],
                         [0.0, 0.55, 0.4, 0.3],
                         [0.0, 0.6, 0.45, 0.5]]

    similarity_matrix = np.asarray(similarity_matrix)

    output = label.match_gale_shapley(labels_x, labels_x, similarity_matrix)

    assert np.array_equal(output, labels_ref)


if __name__ == "__main__":
    test_gale_shapley2()
    test_gale_shapley()
    test_match_labels_3()
    test_match_labels_2()
    test_match_labels()
