# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy


def max_similarity(label_image_x, label_image_y, similarity_matrix):
    """
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

    list_of_men = np.unique(np.append(0, label_image_x.ravel())).astype(np.uint64)
    list_of_women = np.unique(np.append(0, label_image_y.ravel())).astype(np.uint64)

    set_of_labels = np.unique([label_image_x, label_image_y])

    mmax = np.max([list_of_men.max(), list_of_women.max()])

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
            man = {
                'Name': label,
                'Preference': None,
                'Partner': None,
                'Engaged': False
                }
            unmatchable_men.append(man)
            continue

        # Sort list of preferred partners (label) according to preference value
        prefs, partners = sort_lists(non_zero_preferences, non_zero_partners)

        man = {
            'Name': label,
            'Preference': list(partners),
            'Partner': None,
            'Engaged': False
            }
        men.append(man)

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
            woman = {
                    'Name': label,
                    'Preference': None,
                    'Partner': None,
                    'Engaged': False
                    }
            unmatchable_women.append(woman)
            continue

        # Sort partner indeces according to preference score
        prefs, partners = sort_lists(non_zero_preferences, non_zero_partners)

        woman = {
            'Name': label,
            'Preference': list(partners),
            'Partner': None,
            'Engaged': False
            }
        women.append(woman)

    men_by_name = build_dict(men, key="Name")

    # Iterate over women and let them propose to men
    while len(women) > 0:
        w = women[0]
        del women[0]  # remove woman from waiting hall

        # If all possible partners rejected this woman:
        if len(w['Preference']) == 0:
            unmatchable_women.append(w)
            continue

        # Find partner from dictionary
        m = men_by_name.get(w['Preference'][0])

        if m is None:
            print('Here!')

        # If man has no partner yet
        if not m['Engaged']:
            m['Partner'] = w
            w['Partner'] = m
            m['Engaged'], w['Engaged'] = True, True

        else:
            try:
                # Chek if this woman is higher or lower in m's Preference list
                pref_m = np.asarray(m['Preference'], dtype=(np.uint64))
                pos_current_w = np.argwhere(pref_m == m['Partner']['Name']).ravel()[0]
                pos_new_w = np.argwhere(pref_m == w['Name']).ravel()[0]
            except Exception:
                print('Halt')

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

    # Find remaining, unmatched men and delete them from dict of matched men
    remaining_singles = [idx for idx in men_by_name.keys() if men_by_name[idx]['Engaged'] == False]
    if len(remaining_singles) > 0:
        remaining_singles = [men_by_name.pop(idx) for idx in remaining_singles]
        unmatchable_men.append(remaining_singles)

    # Unravel assignment dictionary
    LUT = np.arange(0, mmax+1, 1, dtype=np.uint64)

    # Assign matchable women to new labels
    for idx in men_by_name.keys():
        LUT[men_by_name[idx]['Partner']['Name']] = idx

    # Lastly, assign unmatchable women to new labels
    for w in unmatchable_women:
        LUT[w['Name']] = LUT.max() + 1

    return LUT[label_image_y]


def build_dict(seq, key):
    return dict((d[key], dict(d, index=index)) for (index, d) in enumerate(seq))


def sort_lists(list1, list2):
    if type(list1) == np.ndarray:
        list1 = list1.tolist()

    if type(list2) == np.ndarray:
        list2 = list2.tolist()
    list1, list2 = zip(*sorted(zip(list1, list2)))
    return list1[::-1], list2[::-1]

