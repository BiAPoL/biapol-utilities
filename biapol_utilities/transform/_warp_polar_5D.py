# -*- coding: utf-8 -*-

from skimage.transform import warp_polar
import numpy as np


def warp_polar_5D(image, center=None, radius=None, project=None,
                  scaling='linear', **kwargs):
    """
    Transform an image with up to 5 dimensions into polar coordinates.

    Optionally return axes projections, like cylinder projection.

    Parameters
    ----------
    image : ndarray
        Input image. 2D to 5D images with axis order CTZYX
    center: tuple (row, col), optional
        MPoint in image that represents the center of the transformation
        (i.e., the origin in cartesian space). Values can be of type float.
        If no value is given, the center is assumed to be the center point of
        the image.
    radius: float, optional
        Radius of the circle that bounds the area to be transformed. If no
        value is given, half the minimum between x and y axes is used.
    project{'c', 't', 'z', 'a', 'r', or a combination of those}, optional
        Maximum projection into the specified axes (channel, time, z, angle,
        radius). Number of concomitant projections must be equal or smaller
        than image shape - 2. If no value is given, return all axes.
    scaling{'linear', 'log'}, optional
        Specify whether the image warp is polar or log-polar. Defaults to
        'linear'.


    Returns
    -------
    image_polar : ndarray
        The polar or log-polar warped image or a projection of it.
    """
    if radius is None:
        # Radius is half the smallest image size (x,y)
        radius = min(image.shape[-2], image.shape[-1])//2

    #  Build output with appropriate shape
    other_dims = list(image.shape[:-2])
    other_dims.reverse()
    new_shape = [360, radius]
    for dim in other_dims:
        new_shape.insert(0, dim)
    image_polar = np.zeros(new_shape)

    if len(image.shape) == 2:
        image_polar = warp_polar(image, center=center, radius=radius,
                                 scaling=scaling, **kwargs)

    elif len(image.shape) == 3:
        str2dim = {'z': 0, 'a': 1, 'r': 2}
        for z in range(image.shape[0]):
            image_polar[z] = warp_polar(image[z], center=center, radius=radius,
                                        scaling=scaling, preserve_range=True,
                                        **kwargs)
        if project is not None:
            try:
                if len(project) > 1:
                    print('''Error! Only 1 axis projection allowed for 3D
                          images''')
                    return
                # Return valid projection
                return(np.amax(image_polar, axis=str2dim[project]))
            except KeyError as err:
                print('Error! Unknown ',
                      err,
                      ''' projection. Allowed projections for 3D image are
                       \'z\' for z, \'a\' for angle, \'r\' for radius or
                       None''')
                return

    elif len(image.shape) == 4:
        str2dim = {'t': 0, 'z': 1, 'a': 2, 'r': 3}
        for t in range(image.shape[0]):
            for z in range(image.shape[1]):
                image_polar[t, z] = warp_polar(image[t, z], center=center,
                                               radius=radius, scaling=scaling,
                                               **kwargs)
        if project is not None:
            try:
                dims = []
                for s in project:
                    dims.append(str2dim[s])
                if len(dims) > 2:
                    print('''Error! Only a maximum of 2 axis projection allowed
                          for 4D images''')
                    return
                # Return valid projection
                return(np.amax(image_polar, axis=tuple(dims)))
            except KeyError as err:
                print('Error! Unknown ',
                      err,
                      ''' projection. Allowed projections for 4D image are
                      \'t\' for time, \'z\' for z, \'a\' for angle, \'r\' for
                      radius, a combination of those (like \'tz\' for time and
                                                      z) or None''')
                return

    elif len(image.shape) == 5:
        str2dim = {'c': 0, 't': 1, 'z': 2, 'a': 3, 'r': 4}
        for c in range(image.shape[0]):
            for t in range(image.shape[1]):
                for z in range(image.shape[2]):
                    image_polar[c, t, z] = warp_polar(image[c, t, z],
                                                      center=center,
                                                      radius=radius,
                                                      scaling=scaling,
                                                      **kwargs)
        if project is not None:
            try:
                dims = []
                for s in project:
                    dims.append(str2dim[s])
                if len(dims) > 3:
                    print('''Error! Only a maximum of 3 axis projection allowed
                          for 5D images''')
                    return
                # Return valid projection
                return(np.amax(image_polar, axis=tuple(dims)))
            except KeyError as err:
                print('Error! Unknown ',
                      err,
                      ''' projection. Allowed projections for 5D image are
                      \'c\' for channel, \'t\' for time, \'z\' for z, \'a\'
                      for angle, \'r\' for radius, a combination of those
                      (like \'tz\' for time and z) or None''')
                return
    # Return warped image (same dimensions as input)
    return(image_polar)
