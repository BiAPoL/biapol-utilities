"""Apply function to images owning higher dimensions than designed input."""
import numpy as np
import dask
import dask.array as da


# Adapted from https://stackoverflow.com/a/7663441/11885372
def _get_diff_in_dimensions(func, shape, *args, **kwargs):
    # Same shape as image
    shape = tuple([1]*len(shape))
    dummy_image = np.ones(shape)
    for attempt in range(10):
        try:
            func(dummy_image, *args, **kwargs)
        except Exception as e:
            dummy_image = dummy_image[0]
        else:
            return(attempt)
    return(None)


def _apply_nD_func_to_nplus1_D_image(image, func, *args, **kwargs):
    """Apply function whose input is a nD image to a (n+1) image using dask."""
    # List of lazy (to be) processed images
    list_of_dask = [
                    dask.delayed(func)(image[i],
                                       *args, **kwargs)
                    for i in range(image.shape[0])
    ]
    # List of lazy dask arrays
    dask_arrays = [
        da.from_delayed(lazy_array, shape=image[0].shape, dtype=image[0].dtype)
        for lazy_array in list_of_dask
    ]
    # List to dask stack
    dask_stack = da.stack(dask_arrays, axis=0)
    return(dask_stack)


def process_higher_dimension(image, function, function_input_dimension=None,
                             *args, **kwargs):
    """
    Apply function to images owning higher dimensions than designed input.

    Parameters
    ----------
    image : ndarray
        Input image.
    function : python function
        A function that acts on images and returns images, like those from
        skimage library.
    function_input_dimension : int
        The function maximal accepted input image dimension. By default 'None'.
        In this case, it tries to guess by iteratively applying 'function'
        to a dummy 'image' with decreasing dimensions until it works (maximum
        of 10 reduced image dimensions).


    Returns
    -------
    output : ndarray
        The processed image. If the func has other outputs, returns them
        according to func definition.
    """
    # Get difference between image dimension and accepted function input
    if function_input_dimension is not None:
        diff_dimensions = len(image.shape) - function_input_dimension
    else:
        # Tries to find the difference in dimensions
        diff_dimensions = _get_diff_in_dimensions(function, image.shape,
                                                  *args, **kwargs)
    if diff_dimensions is None:
        print('Failed to match function input and image.')
        return
    elif diff_dimensions > 0:
        params = [image]
        for i in range(diff_dimensions-1):
            params.append(_apply_nD_func_to_nplus1_D_image)
        params.append(function)
        # Applies function 'func' on image 'diff_dimensions - 1' times
        return(_apply_nD_func_to_nplus1_D_image(*params, *args, **kwargs))
    else:
        # If func can be applied directly to image, do so
        return(function(image, *args, **kwargs))
