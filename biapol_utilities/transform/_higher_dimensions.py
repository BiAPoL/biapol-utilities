"""Apply function to images owning higher dimensions than designed input."""
import numpy as np
import dask
import dask.array as da


# Adapted from https://stackoverflow.com/a/7663441/11885372
def _get_diff_in_dimensions(func, shape, max_attempts=10, *args, **kwargs):
    """Get difference between image dimension and function accepted input."""
    # Same shape as image
    dummy_image = np.ones(shape)
    for attempt in range(max_attempts):
        print(attempt)
        try:
            func(dummy_image, *args, **kwargs)
        except Exception as e:
            print(str(e))
            dummy_image = dummy_image[0]
        else:
            print('Success after ', attempt, 'reduced dimension(s)')
            return(attempt)
    print('Failed')
    return(None)


def _apply_nD_func_to_nplus1_D_image(image, func, *args, **kwargs):
    """Apply function whose input is a nD image to a (n+1) image using dask."""
    @dask.delayed
    def read_first_axis(image, i):
        image_nminus1_D = image[i]
        return(image_nminus1_D)

    # List of lazy (to be) processed images
    list_of_dask = [
                    dask.delayed(func)(read_first_axis(image, i),
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


def process_higher_dimension(image, func, *args, **kwargs):
    """
    Apply function to images owning higher dimensions than designed input.

    Parameters
    ----------
    image : ndarray
        Input image.
    func : python function
        A function that acts on images and returns images, like those from
        skimage library.


    Returns
    -------
    output : ndarray
        The processed image. If the func has other outputs, returns them
        according to func definition.
    """
    # Get difference between image dimension and accepted function input
    diff_dimensions = _get_diff_in_dimensions(func, image.shape,
                                              *args, **kwargs)
    if diff_dimensions > 0:
        params = [image]
        for i in range(diff_dimensions-1):
            params.append(_apply_nD_func_to_nplus1_D_image)
        params.append(func)
        # Applies function 'func' on image 'diff_dimensions - 1' times
        return(_apply_nD_func_to_nplus1_D_image(*params, *args, **kwargs))
    else:
        # If func can be applied directly to image, do so
        return(func(image, *args, **kwargs))
