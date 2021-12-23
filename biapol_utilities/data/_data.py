# -*- coding: utf-8 -*-

from skimage import io
import numpy as np
import matplotlib
import os
from skimage.morphology import disk

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
    """Hourglass 3D+t image generator.


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
    """

    def __init__(self, radius=100, glass_value=100, content_value=200):
        self.y = 2*radius
        self.x = 2*radius
        self.z = 2*radius
        self.time = radius//2
        self.shape = (self.y, self.x)
        self.glass_value = glass_value
        self.content_value = content_value
        self._create_image()

    def _create_disk(self, radius, fill=False):
        # Create a filled or hollow disk image
        filled_disk = disk(radius)
        if fill is True:
            return(filled_disk)
        else:
            filled_disk_in = np.pad(disk(radius - 1), ((1, 1), (1, 1)))
            return(filled_disk - filled_disk_in)

    def _pad2shape(self, image, shape):
        #  Pad image with zeros until it has desired shape
        padding_y = shape[0] - image.shape[0]
        padding_x = shape[1] - image.shape[1]
        if ((padding_y % 2) == 0) & ((padding_x % 2) == 0):
            padding = ((padding_y//2, padding_y//2), (padding_x//2, padding_x//2))
        elif ((padding_y % 2) == 0) & ((padding_x % 2) != 0):
            padding = ((padding_y//2, padding_y//2), ((padding_x//2) + 1, padding_x//2))
        elif ((padding_y % 2) != 0) & ((padding_x % 2) == 0):
            padding = (((padding_y//2) + 1, padding_y//2), (padding_x//2, padding_x//2))
        else:
            padding = (((padding_y//2) + 1, padding_y//2), ((padding_x//2) + 1, padding_x//2))
        return(np.pad(image, padding))

    def _create_image(self):
        # Create hourglass image
        self.image = np.zeros((self.time, self.z, self.y, self.x),
                              dtype='uint8')
        lower_content_radius = 1
        radius_step = 1
        for t in range(self.time):
            glass_radius = self.y // 2
            upper_content_radius = self.z // 4
            for k in range(self.z):
                # Create hourglass structure
                if ((k == 1) | (k == self.z - 1)):
                    fill = True  # Bottom and top are filled
                else:
                    fill = False
                if k != 0:
                    #  Draw circular glass wall
                    disk = self._create_disk(glass_radius, fill)
                    disk = self._pad2shape(disk, self.shape)
                    self.image[t, k] =  disk * self.glass_value
                #  Continuously decrease glass radius while in upper part
                if k < ((self.z // 2) - 1):
                    glass_radius -= radius_step
                elif k == ((self.z // 2) - 1):
                    glass_radius = 1
                elif k == (self.z // 2):
                    glass_radius = radius_step
                #  Continuously increase glass radius while in lower part
                else:
                    glass_radius += radius_step
                #  Add upper content (from quarter to half image)
                if ((k >= self.z // 4) & (k < self.z // 2)):
                    disk = self._create_disk(upper_content_radius, True)
                    disk = self._pad2shape(disk, self.shape)
                    self.image[t, k] += disk * self.content_value
                    upper_content_radius -= radius_step

            #  Add lower content: arc droplet
            if t == 0:
                self.image[t,
                            self.z // 2,
                            self.y // 2,
                            self.x // 2] = self.content_value
            if t >= 1:
                half_disk = self._create_disk(lower_content_radius, False)
                half_disk[:, :half_disk.shape[1] // 2] = 0
                half_disk = self._pad2shape(half_disk, self.shape)
                half_disk *= self.content_value
                self.image[t, ((self.z // 2) + 2*t)] += half_disk
                lower_content_radius += 2*radius_step


def hourglass():
    """Gray-level 3D+t hourglass image.

    Can be used for polar transformations, tracking, dimentionality reduction.

    Returns
    -------
    hourglass : (50, 200, 200, 200) uint8 ndarray
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
