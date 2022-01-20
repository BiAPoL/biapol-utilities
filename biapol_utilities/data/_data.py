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
        Hourglass maximum radius value. Default value: 100. Minimum value: 1.
        It also defines image shape as (radius, 2*radius + 1, 2*radius + 1,
                                        2*radius + 1).
    glass_value: uint8, optional
        Voxel intensity value of the hourglass walls.
    content_value: uint8, optional
        Voxel intensity value of the hourglass contents.
    image_shape: 4-tuple, optional
        Shape of desired resulting image. x, y and z must be equal. Minimum
        value: (1,3,3,3).
    pad_edges: bool, optional
        If True, pads image with 1 voxel wide border. This increases each
        spatial axes size by 2. Default: False.

    """

    def __init__(self, radius=100, glass_value=100, content_value=200,
                 image_shape=None, pad_edges=False, *args, **kwargs):
        # Check minimum size
        if (radius < 1):
            print('Error! Radius too small! Minimum radius = 1.')
            return
        # Check proper shape (x, y, z must be equal)
        if image_shape is not None:
            if ((image_shape[0] < 1) or (image_shape[-1] < 3)):
                print('Error! Image size too small! Minimum shape = (1,3,3,3)')
                return
            if image_shape[1] == image_shape[2] == image_shape[3]:
                self.radius = int((image_shape[3] - 1) / 2)
                self.time = self.radius  # time = r
                self.time_factor = image_shape[0]/self.time
            else:
                print('Error! xyz dimensions must be equal.')
                return
        else:
            self.radius = radius
            self.time_factor = 1

        self.y = 2 * self.radius + 1  # y = 2r + 1
        self.x = 2 * self.radius + 1  # x = 2r + 1
        self.z = 2 * self.radius + 1  # z = 2r + 1
        self.time = self.radius  # time = r
        self.time *= self.time_factor
        self.time = round(self.time)

        self.shape_yx = (self.y, self.x)
        self.glass_value = glass_value
        self.content_value = content_value
        self._create_image()

        # Match desired shape spatial dimensions
        if image_shape is not None:
            if self.image.shape[-1] < image_shape[-1]:
                # if desired shape is even, pads image
                self.image = np.pad(self.image, ((0, 0),
                                                 (1, 0),
                                                 (1, 0),
                                                 (1, 0)))

        # Pad image with 1 pixel wide border
        if pad_edges:
            # pad edges for better viewing
            self.image = np.pad(self.image, ((0, 0),
                                             (1, 1),
                                             (1, 1),
                                             (1, 1)))

    def _create_disk(self, radius, fill=False):
        # Create a filled or hollow disk image
        filled_disk = disk(radius)
        if fill is True:
            return(filled_disk)
        else:
            filled_disk_in = np.pad(disk(radius - 1), ((1, 1), (1, 1)))
            return(filled_disk - filled_disk_in)

    def _pad2shape(self, image, shape_yx):
        #  Pad 2D image with zeros until it has desired shape
        padding_y = shape_yx[0] - image.shape[0]
        padding_x = shape_yx[1] - image.shape[1]
        if ((padding_y % 2) == 0) & ((padding_x % 2) == 0):
            padding = ((padding_y//2, padding_y//2),
                       (padding_x//2, padding_x//2))
        elif ((padding_y % 2) == 0) & ((padding_x % 2) != 0):
            padding = ((padding_y//2, padding_y//2),
                       ((padding_x//2) + 1, padding_x//2))
        elif ((padding_y % 2) != 0) & ((padding_x % 2) == 0):
            padding = (((padding_y//2) + 1, padding_y//2),
                       (padding_x//2, padding_x//2))
        else:
            padding = (((padding_y//2) + 1, padding_y//2),
                       ((padding_x//2) + 1, padding_x//2))
        return(np.pad(image, padding))

    def _create_image(self):
        # Create hourglass image
        self.image = np.zeros((self.time, self.z, self.y, self.x),
                              dtype='uint8')
        radius_step = 1
        for t in range(self.time):
            glass_radius = self.radius
            for k in range(self.z):
                # Create hourglass structure
                if ((k == 0) | (k == (self.z - 1))):
                    fill = True  # Bottom and top are filled
                else:
                    fill = False
                #  Draw circular glass wall
                disk = self._create_disk(glass_radius, fill)
                # Match image shape
                disk = self._pad2shape(disk, self.shape_yx)
                self.image[t, k] = disk * self.glass_value

                #  Add upper content (from quarter to half image)
                if ((k >= self.z // 4) & (k < self.z // 2)):
                    disk = self._create_disk(glass_radius - 1, True)
                    disk = self._pad2shape(disk, self.shape_yx)
                    self.image[t, k] += (disk * self.content_value)

                #  Continuously decrease glass radius while in upper part
                if ((k != 0) & (k < (self.z // 2))):
                    glass_radius -= radius_step
                #  Continuously increase glass radius while in lower part
                elif ((k is not (self.z - 2)) & (k >= (self.z // 2))):
                    glass_radius += radius_step

            #  Add arc droplet
            if t/self.time_factor < 1:
                # Add bottleneck point
                self.image[t,
                           self.z // 2,
                           self.y // 2,
                           self.x // 2] = self.content_value
            else:
                half_disk = self._create_disk(int(t/self.time_factor), False)
                half_disk[:, :half_disk.shape[1] // 2] = 0
                half_disk = self._pad2shape(half_disk, self.shape_yx)
                half_disk *= self.content_value
                self.image[t,
                           ((self.z // 2)
                            + int(t/self.time_factor))] += half_disk


def hourglass(*args, **kwargs):
    """Gray-level 3D+t hourglass image.

    Can be used for polar transformations, tracking, dimentionality reduction.

    Parameters
    ----------
    radius : int, optional
        Hourglass maximum radius value. Default value: 100. Minimum value: 1.
        It also defines image shape as (radius, 2*radius + 1, 2*radius + 1,
                                        2*radius + 1).
    glass_value: uint8, optional
        Voxel intensity value of the hourglass walls.
    content_value: uint8, optional
        Voxel intensity value of the hourglass contents.
    image_shape: 4-tuple, optional
        Shape of desired resulting image. x, y and z must be equal. Minimum
        value: (1,3,3,3).
    pad_edges: bool, optional
        If True, pads image with 1 voxel wide border. This increases each
        spatial axes size by 2. Default: False.

    Returns
    -------
    hourglass : (100, 201, 201, 201) uint8 ndarray
        Hourglass 3D+t (time, z, y, x) image depicting an expanding bright
        half-circle droplet sliding through the glass wall.
    """
    return Hourglass(*args, **kwargs).image


def labels_colormap():
    if not hasattr(labels_colormap, "labels_cmap"):
        state = np.random.RandomState(1234567890)

        lut = state.rand(65537, 3)
        lut[0, :] = 0
        labels_colormap.labels_cmap = matplotlib.colors.ListedColormap(lut)
    return labels_colormap.labels_cmap
