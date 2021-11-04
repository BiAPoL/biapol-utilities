# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 15:40:27 2021

@author: johan
"""

from sklearn import metrics
import numpy as np

def label_confusion_matrix(labels_true, labels_pred):
    
    
    labels_true = labels_true.flatten()
    labels_pred = labels_pred.flatten()