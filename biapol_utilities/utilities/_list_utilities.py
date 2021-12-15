# -*- coding: utf-8 -*-

import numpy as np


def sort_list_pairs(list1, list2, **kwargs):
    """
    Sort numeric list according to values in a second array.

    This function orders the values in `list2` according to the values passed
    as `list1`. The contents can be every sortable format.
    Code adapted from [0]

    Parameters
    ----------
    list1 : list
        List of values that serve as a reference
    list2 : list
        List of values that are sorted according to list1.

    order: str, optional
        Parameter determines whether returned lists are sorted to
        descending/ascending order of values in `list1`. The default value is
        "descending".

    Returns
    -------
    list
        Sorted `list1`.
    list
        `list2` sorted according to values in `list1`.

    Reference
    -------
    .. [1] https://stackoverflow.com/a/9764364
    """
    order = kwargs.get('order', 'descending')

    if type(list1) == np.ndarray:
        list1 = list1.tolist()

    if type(list2) == np.ndarray:
        list2 = list2.tolist()
    list1, list2 = zip(*sorted(zip(list1, list2)))

    if order == 'descending':
        return list1[::-1], list2[::-1]
    elif order == 'ascending':
        return list1, list2
