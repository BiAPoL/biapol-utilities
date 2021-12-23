# -*- coding: utf-8 -*-

from skimage import io
import numpy as np
import matplotlib
import os

data_dir = os.path.abspath(os.path.dirname(__file__))


def blobs():
    """Gray-level "blobs" image [1].

    Can be used for segmentation and denoising examples.

    Returns
    -------
    blobs : (256, 254) uint8 ndarray
        Blobs image.

    References
    ----------
    .. [1] https://imagej.nih.gov/ij/images/
    """
    return io.imread(os.path.join(data_dir, "blobs.png"))


class Hourglass:
    """Hourglass object [1].

    Can be used to generate a Gray-level (8-bit) 3D+t image.

    Attributes
    ----------
    radius : int, optional
        Maximum radius value. It also defines image shape as (radius/2, radius,
        2*radius, 2*radius)
    glass_value: uint8, optional
        Voxel intensity value of the hourglass walls.
    content_value: uint8, optional
        Voxel intensity value of the hourglass contents.
    References
    ----------
    .. [1] Created by Marcelo Leomil Zoccoler
    """

    def __init__(self, radius=100, glass_value=100, content_value=200):
        self.lin = 2*radius
        self.col = 2*radius
        self.dep = radius
        self.time = radius//2
        self.shape = (self.lin, self.col)
        self.center = (self.shape[0]//2, self.shape[1]//2)
        self.glass_value = glass_value
        self.content_value = content_value
        self._create_image()

    def _create_circle(self, radius, fill=False):
        # Create a filled or hollow circle image
        grid = np.meshgrid(range(self.shape[0]), range(self.shape[1]))
        circle = (grid[0] - self.center[0])**2 + (grid[1] - self.center[1])**2
        if fill is False:
            circle = (circle <= radius**2) & (circle > (radius-1)**2)
        else:
            circle = circle <= radius**2
        return(circle.astype('uint8'))

    def _create_image(self):
        # Create hourglass image
        self.image = np.zeros((self.time, self.dep, self.lin, self.col),
                              dtype='uint8')
        content_radius_down = 1
        for t in range(self.time):
            radius = self.lin // 2
            content_radius_up = self.dep // 2
            for k in range(self.dep):
                # Create hourglass structure
                if ((k == 1) | (k == self.dep - 1)):
                    fill = True
                else:
                    fill = False
                if k != 0:
                    self.image[t, k] = self._create_circle(radius, fill) *\
                        self.glass_value
                if k < ((self.dep // 2) - 1):
                    radius -= 2
                elif k == ((self.dep // 2) - 1):
                    radius = 1
                elif k == (self.dep // 2):
                    radius = 2
                else:
                    radius += 2
                # Add upper content
                if ((k >= self.dep // 4) & (k < self.dep // 2)):
                    self.image[t, k] += self._create_circle(content_radius_up,
                                                            True) *\
                        self.content_value

                    content_radius_up -= 2
            # Add lower arc droplet
            if t == 1:
                self.image[t,
                           self.dep // 2,
                           self.lin // 2,
                           self.col // 2] = self.content_value
            if t > 1:
                half_circle = self._create_circle(content_radius_down,
                                                  False) *\
                    self.content_value

                half_circle[:, :half_circle.shape[1] // 2] = 0
                self.image[t, ((self.dep // 2) - 1 + t)] += half_circle
                content_radius_down += 2


def hourglass():
    """Gray-level 3D+t hourglass image.

    Can be used for polar transformations, tracking, dimentionality reduction.

    Returns
    -------
    hourglass : (50, 100, 200, 200) uint8 ndarray
        Hourglass 3D+t (time, z, y, x) image depicting an expanding bright
        half-circle droplet sliding through the glass wall.
    """
    return Hourglass().image


def labels_colormap():
    if not hasattr(labels_colormap, "labels_cmap"):
        state = np.random.RandomState(1234567890)

        lut = state.rand(65537, 3)
        lut[0, :] = 0
        labels_colormap.labels_cmap = matplotlib.colors.ListedColormap(lut)
    return labels_colormap.labels_cmap
