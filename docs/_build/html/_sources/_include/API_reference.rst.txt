API Reference
=============
This is the class and function reference of biapol-utilities. Please refer to the examples for further details, as the class and function raw specifications may not be enough to give full guidelines on their uses.

The label module
------------------
The label module bundles functions that operate on label images, i.e. images with annotated objects. The idendity of
an object is encoded in the pixel value which must be an integer number.

.. currentmodule:: biapol_utilities

.. autosummary::
    :recursive:
    :toctree: generated

    label.compare_labels
    label.jaccard_index_matrix
    label.match_labels
    label.match_labels_stack

The data module
------------------
The data module provides example data to be usedin the examples section.

.. currentmodule:: biapol_utilities

.. autosummary::
    :recursive:
    :toctree: generated

    data.blobs