# -*- coding: utf-8 -*-

import numpy as np
from biapol_utilities import utilities


def max_similarity(label_image_x, label_image_y, similarity_matrix):
    """
    Maximum-similarity algorithm for label-matching.

    Matches labels in two input label images (label_image_x and label_image_y)
    based on the maximal value in the similarity_matrix.

    Parameters
    ----------
    label_image_x : ND-array, int
        labelled input image which serves as reference image
    label_image_y : ND-array, int
        Labbelled input image, the labels of which will be matched up with the
        labels in label_image_x based on the chosen input metric
    similarity_matrix : 2D-array, float
        matrix containing the metric that defines the similarity of labels
        in label_image_x and label_image_y, with axis=0 referring to the labels
        of label_image_x. In other words, the entry similarity_matrix[i, j]
        corresponds to the "similarity" between label i in label_image_x and
        label j in label_image_y.

    Returns
    -------
    ND-array
        Relabelled version of input_image_y. Unmatched labels are appended to
        the set of matched labels and thus, the total number of labels may
        increase.

    """
    # Suppress non-maximal entries
    similarity_matrix[similarity_matrix < similarity_matrix.max(axis=0)] = 0.0

    mmax = label_image_x.max()

    if similarity_matrix.size > 0:
        # Pick value with highest IoU value
        istitch = similarity_matrix.argmax(axis=1) + 1

        # Find unpaired labels
        ino = np.nonzero(similarity_matrix.max(axis=1) == 0.0)[0]

        # append unmatched labels and background to lookup table
        istitch[ino] = np.arange(mmax+1, mmax+len(ino)+1, 1, int)
        mmax += len(ino)
        istitch = np.append(np.array(0), istitch)

        return istitch[label_image_y]

    else:
        raise ValueError('Similarity matrix dimension was expected to have '
                         'size MxM but was found to have size '
                         f'{similarity_matrix.size}. Check calculation of'
                         'similarity matrix for errors.')


def gale_shapley(label_image_x, label_image_y, similarity_matrix):
    """
    Gale-Shapley stable-marriage algorithm for label-matching.

    Implementation of Gale-Shapley's solution of the stable-marriage problem
    [0, 1] to match labels from two label images (`label_image_x` and
    `label_image_y`) based on a mutual set of preferences that are passed as
    `similarity_matrix`. Unmatchable labels in `label_image_x` maintain their
    previous labels whereas unmatchable labels in `label_image_y` may be
    assigned to new labels, thus potentially increasing the total number of
    labels.

    Parameters
    ----------
    label_image_x : ND-array, int
        label image that serves as reference label image
    label_image_y : ND-array, int
        Label image, the labels of which are assigned to `label_image_x`
    similarity_matrix : NxN, float
        similarity matrix where `similarity_matrix[i, j]` corresponds to the
        similarity/preference to match the i-th and j-th entry from the set of
        labels `set(label_image_x, label_image_y)` with each other.

    Returns
    -------
    ND-array, int
        Copy of label_image_y with labels assigned to labels_image_x.

    References
    ----------
    .. [0] https://en.wikipedia.org/wiki/Gale%E2%80%93Shapley_algorithm
    .. [1] https://doi.org/10.1080/00029890.1962.11989827

    """
    # Get set of input labels from both datasets and combined labels
    list_of_men = np.unique(np.append(0, label_image_x.ravel()))
    list_of_women = np.unique(np.append(0, label_image_y.ravel()))
    set_of_labels = np.unique([label_image_x, label_image_y])

    # Highest label found
    mmax = np.max([list_of_men.max(), list_of_women.max()])

    # Allocate lists for unmatchable labels
    unmatchable_men = []
    unmatchable_women = []

    # Create a list with 'man' entries, whereas each entry is a dict with
    # the man's Name, his marital partner and his preferences.
    men = []

    # list of man refers to label, entry-1 refers to respective entry in
    # similarity matrix
    for idx, label in enumerate(set_of_labels):

        # is this label among set of labels_x?
        if not any(list_of_men == label):
            continue

        # get array with preferences
        preference = similarity_matrix[:, idx]

        # Get partner indeces
        non_zero_preferences = preference[preference != 0]
        non_zero_partners = set_of_labels[preference != 0]

        # If label is among set of labels_x, but has no match in label_y
        if len(non_zero_partners) == 0:
            unmatchable_men.append({
                'Name': label,
                'Preference': None,
                'Partner': None,
                'Engaged': False
                })
            continue

        # Sort list of preferred partners (label) according to preference value
        prefs, partners = utilities.sort_list_pairs(non_zero_preferences,
                                                    non_zero_partners)

        men.append({
            'Name': label,
            'Preference': list(partners),
            'Partner': None,
            'Engaged': False
            })

    # Repeat with women
    women = []
    for idx, label in enumerate(set_of_labels):

        # is this label among set of labels_y?
        if not any(list_of_women == label):
            continue

        # get array with preferences
        preference = similarity_matrix[idx, :]

        # Get partner indeces
        non_zero_preferences = preference[preference != 0]
        non_zero_partners = set_of_labels[preference != 0]

        # If label is among set of labels_y, but has no match in label_x
        if len(non_zero_partners) == 0:
            unmatchable_women.append({
                'Name': label,
                'Preference': None,
                'Partner': None,
                'Engaged': False
                })
            continue

        # Sort partner indeces according to preference score
        prefs, partners = utilities.sort_list_pairs(non_zero_preferences,
                                                    non_zero_partners)

        women.append({
            'Name': label,
            'Preference': list(partners),
            'Partner': None,
            'Engaged': False
            })

    # Make dictionary that assigns `man["Name"]` to his index in list `men`
    men_by_name = dict(
        (d["Name"], dict(d, index=index)) for (index, d) in enumerate(men)
        )

    # Iterate over women and let them propose to men
    while len(women) > 0:
        w = women[0]
        del women[0]  # remove woman from queue

        # If all possible partners rejected this woman, i.e.
        # this woman has zero remaining preferences:
        if len(w['Preference']) == 0:
            unmatchable_women.append(w)
            continue

        # Find partner from dictionary of men
        m = men_by_name.get(w['Preference'][0])

        # If man has no partner yet
        if not m['Engaged']:
            m['Partner'] = w
            w['Partner'] = m
            m['Engaged'], w['Engaged'] = True, True

        # If man has partner
        else:

            # Check if this woman is higher or lower in m's Preference list
            pref_m = np.asarray(m['Preference'], dtype=(np.uint64))
            pos_current_w = np.argwhere(pref_m == m['Partner']['Name']).ravel()[0]
            pos_new_w = np.argwhere(pref_m == w['Name']).ravel()[0]

            # Check who is further up on m's preference list
            if pos_new_w > pos_current_w:

                # If w is lower on man's list than current partner:
                # Remove this man from women's list and add to queue again
                del w['Preference'][0]
                women.append(w)
            else:

                # Remove current partner from this man and add to queue
                w_rejected = m['Partner']
                w_rejected['Engaged'] = False
                w_rejected['Partner'] = None
                women.append(w_rejected)

                # Engage m and w
                m['Partner'] = w
                w['Partner'] = m
                m['Engaged'], w['Engaged'] = True, True

    # Find remaining, unmatched men and delete them from dict of men
    # Thus, only matched up men and their respective partners are in this dict.
    remaining_singles = [idx for idx in men_by_name.keys() if not men_by_name[idx]['Engaged']]
    if len(remaining_singles) > 0:
        remaining_singles = [men_by_name.pop(idx) for idx in remaining_singles]
        unmatchable_men.append(remaining_singles)

    # Create a look-up table that assigns labels in label_image_y (aka women)
    # to labels in label_image_x (aka men)
    LUT = np.arange(0, mmax+1, 1, dtype=np.uint64)

    # Assign matchable women to new labels
    for idx in men_by_name.keys():
        LUT[men_by_name[idx]['Partner']['Name']] = idx

    # Lastly, assign unmatchable women to new labels
    for w in unmatchable_women:
        LUT[w['Name']] = LUT.max() + 1

    return LUT[label_image_y]
