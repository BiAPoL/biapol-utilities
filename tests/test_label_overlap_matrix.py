# -*- coding: utf-8 -*-

import biapol_utilities as bputils
import numpy as np


def test_label_overlap_matrix():
    
    # create test arrays
    a = np.asarray([[1, 1, 0],
                    [1, 1, 2],
                    [0, 2, 2]])
    
    b = np.asarray([[1, 1, 0],
                    [1, 2, 2],
                    [0, 2, 2]])
    
    
    overlap = bputils.label.label_overlap_matrix_numpy(a, b)
    
    correct_result = np.asarray([[2, 0, 0],
                                 [0, 3, 1],
                                 [0, 0, 3]])
    
    assert(np.array_equal(overlap, correct_result))
    
def test_label_overlap_matrix2():
    
    # create test arrays
    a = np.asarray([[1, 1, 0],
                    [1, 1, 2],
                    [0, 2, 3]])
    
    b = np.asarray([[1, 1, 0],
                    [1, 2, 2],
                    [0, 2, 2]])
    
    
    overlap = bputils.label.label_overlap_matrix_numpy(a, b)
    
    correct_result = np.asarray([[2, 0, 0],
                                 [0, 3, 1],
                                 [0, 0, 2],
                                 [0, 0, 1]])
    
    assert(np.array_equal(overlap, correct_result))
    
def test_confusion_matrix():
    
    # create test arrays
    a = np.asarray([[1, 1, 0],
                    [1, 1, 2],
                    [0, 2, 2]])
    
    b = np.asarray([[1, 1, 0],
                    [1, 2, 2],
                    [0, 2, 2]])
    
    
    overlap = bputils.label.label_overlap_matrix(a, b)
    
    correct_result = np.asarray([[2, 0, 0],
                                 [0, 3, 1],
                                 [0, 0, 3]])
    
    assert(np.array_equal(overlap, correct_result))
    
def test_confusion_matrix2():
    
    # create test arrays
    a = np.asarray([[1, 1, 0],
                    [1, 1, 2],
                    [0, 2, 3]])
    
    b = np.asarray([[1, 1, 0],
                    [1, 2, 2],
                    [0, 2, 2]])
    
    
    overlap = bputils.label.label_overlap_matrix(a, b)
    
    correct_result = np.asarray([[2, 0, 0],
                                 [0, 3, 1],
                                 [0, 0, 2],
                                 [0, 0, 1]])
    
    assert(np.array_equal(overlap, correct_result))