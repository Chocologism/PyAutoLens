from src.imaging import scaled_array
from src.imaging import grids
from src import exc
import numpy as np
from collections import namedtuple
from functools import wraps
import inspect

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)


class Memoizer(object):
    def __init__(self):
        self.results = {}
        self.calls = 0
        self.arg_names = None

    def __call__(self, func):
        if self.arg_names is not None:
            raise AssertionError("Instantiate a new Memoizer for each function")
        self.arg_names = inspect.getfullargspec(func).args

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = ", ".join(
                ["('{}', {})".format(arg_name, arg) for arg_name, arg in
                 list(zip(self.arg_names, args)) + [(k, v) for k, v in kwargs.items()]])
            if key not in self.results:
                self.calls += 1
            self.results[key] = func(*args, **kwargs)
            return self.results[key]

        return wrapper


class Mask(scaled_array.ScaledArray):
    """
    A mask represented by an ndarray where True is masked.
    """

    @classmethod
    def empty_for_shape_arc_seconds_and_pixel_scale(cls, shape_arc_seconds, pixel_scale):
        return cls(np.full(tuple(map(lambda d: int(d / pixel_scale), shape_arc_seconds)), True), pixel_scale)

    @classmethod
    def circular(cls, shape_arc_seconds, pixel_scale, radius_mask, centre=(0., 0.)):
        """
        Setup the mask as a circle, using a specified arc second radius.

        Parameters
        ----------
        shape_arc_seconds: (float, float)
            The (x,y) image_shape
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        radius_mask : float
            The radius of the circular mask in arc seconds.
        centre: (float, float)
            The centre of the mask.
        """

        grid = Mask.empty_for_shape_arc_seconds_and_pixel_scale(shape_arc_seconds, pixel_scale)

        for x in range(int(grid.shape[0])):
            for y in range(int(grid.shape[1])):
                x_arcsec, y_arcsec = grid.pixel_coordinates_to_arc_second_coordinates((x, y))

                x_arcsec -= centre[0]
                y_arcsec -= centre[1]

                radius_arcsec = np.sqrt(x_arcsec ** 2 + y_arcsec ** 2)

                grid[x, y] = radius_arcsec > radius_mask

        return cls(grid, pixel_scale)

    @classmethod
    def annular(cls, shape_arc_seconds, pixel_scale, inner_radius_mask, outer_radius_mask, centre=(0., 0.)):
        """
        Setup the mask as a circle, using a specified inner and outer radius in arc seconds.

        Parameters
        ----------
        shape_arc_seconds : (float, float)
            The (x,y) image_shape of the mask
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        inner_radius_mask : float
            The inner radius of the annulus mask in arc seconds.
        outer_radius_mask : float
            The outer radius of the annulus mask in arc seconds.
        centre: (float, float)
            The centre of the mask.
        """

        grid = Mask.empty_for_shape_arc_seconds_and_pixel_scale(shape_arc_seconds, pixel_scale)

        for x in range(int(grid.shape[0])):
            for y in range(int(grid.shape[1])):
                x_arcsec, y_arcsec = grid.pixel_coordinates_to_arc_second_coordinates((x, y))

                x_arcsec -= centre[0]
                y_arcsec -= centre[1]

                radius_arcsec = np.sqrt(x_arcsec ** 2 + y_arcsec ** 2)

                grid[x, y] = radius_arcsec > outer_radius_mask or radius_arcsec < inner_radius_mask

        return cls(grid, pixel_scale)

    @classmethod
    def unmasked(cls, shape_arc_seconds, pixel_scale):
        """
        Setup the mask such that all values are unmasked, thus corresponding to the entire image.

        Parameters
        ----------
        shape_arc_seconds : (float, float)
            The (x,y) image_shape of the mask
        pixel_scale: float
            The arc-second to pixel conversion factor of each pixel.
        """
        grid = Mask.empty_for_shape_arc_seconds_and_pixel_scale(shape_arc_seconds, pixel_scale)
        return cls(np.ma.make_mask_none(grid.shape), pixel_scale)

    @classmethod
    def for_simulate(cls, shape_arc_seconds, pixel_scale, psf_size):

        if psf_size[0] % 2 == 0 or psf_size[1] % 2 == 0 or psf_size[0] != psf_size[1]:
            raise exc.KernelException("PSF Kernel must be odd and square")

        ma = cls.unmasked(shape_arc_seconds, pixel_scale)
        pad_size = (int(psf_size[0] / 2) + 1, int(psf_size[1] / 2) + 1)
        return ma.pad(new_dimensions=(ma.shape[0] + pad_size[0], ma.shape[1] + pad_size[1]), pad_value=1)

    @property
    def pixels_in_mask(self):
        return int(np.size(self) - np.sum(self))

    @property
    def coordinate_grid(self):
        """
        Compute the image grid_coords grids from a mask, using the center of every unmasked pixel.
        """
        coordinates = self.grid_coordinates

        pixels = self.pixels_in_mask

        grid = np.zeros(shape=(pixels, 2))
        pixel_count = 0

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if not self[x, y]:
                    grid[pixel_count, :] = coordinates[x, y]
                    pixel_count += 1

        return grids.CoordinateGrid(grid)

    def sub_to_image_with_size(self, grid_size_sub):
        """ Compute the pairing of every sub-pixel to its original image pixel from a mask.

        Parameters
        ----------
        grid_size_sub : int
            The (sub_grid_size x sub_grid_size) of the sub-grid_coords of each image pixel.
        """

        sub_to_image = np.zeros(shape=(self.pixels_in_mask * grid_size_sub ** 2,), dtype=int)
        image_pixel_count = 0
        sub_pixel_count = 0

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if not self[x, y]:
                    for x1 in range(grid_size_sub):
                        for y1 in range(grid_size_sub):
                            sub_to_image[sub_pixel_count] = image_pixel_count
                            sub_pixel_count += 1

                    image_pixel_count += 1

        return sub_to_image

    def masked_1d_array_from_2d_array(self, grid_data):
        """Compute a data grid, which represents the data values of a data-set (e.g. an image, noise, in the mask.

        Parameters
        ----------
        grid_data: ndarray | float | None

        """
        if grid_data is None or isinstance(grid_data, float):
            return grid_data

        pixels = self.pixels_in_mask

        grid = np.zeros(shape=pixels)
        pixel_count = 0

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if not self[x, y]:
                    grid[pixel_count] = grid_data[x, y]
                    pixel_count += 1

        return grid

    def grid_to_pixel(self):
        """
        Compute the mapping of every pixel in the mask to its 2D pixel coordinates.
        """
        pixels = self.pixels_in_mask

        grid = np.zeros(shape=(pixels, 2), dtype='int')
        pixel_count = 0

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if not self[x, y]:
                    grid[pixel_count, :] = x, y
                    pixel_count += 1

        return grid

    def __getitem__(self, coords):
        try:
            return super(Mask, self).__getitem__(coords)
        except IndexError:
            return True

    @property
    def border_pixel_indices(self):
        """Compute the border image data_to_pixels from a mask, where a border pixel is a pixel inside the mask but on
        its edge, therefore neighboring a pixel with a *True* value.
        """

        border_pixels = np.empty(0)
        image_pixel_index = 0

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if not self[x, y]:
                    if self[x + 1, y] or self[x - 1, y] or self[x, y + 1] or \
                            self[x, y - 1] or self[x + 1, y + 1] or self[x + 1, y - 1] \
                            or self[x - 1, y + 1] or self[x - 1, y - 1]:
                        border_pixels = np.append(border_pixels, image_pixel_index)

                    image_pixel_index += 1

        return border_pixels

    @Memoizer()
    def blurring_mask_for_kernel_shape(self, kernel_shape):
        """Compute the blurring mask, which represents all data_to_pixels not in the mask but close enough to it that a
        fraction of their light will be blurring in the image.

        Parameters
        ----------
        kernel_shape : (int, int)
           The sub_grid_size of the psf which defines the blurring region (e.g. the shape of the PSF)
        """

        if kernel_shape[0] % 2 == 0 or kernel_shape[1] % 2 == 0:
            raise exc.MaskException("psf_size of exterior region must be odd")

        blurring_mask = np.full(self.shape, True)

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if not self[x, y]:
                    for y1 in range((-kernel_shape[1] + 1) // 2, (kernel_shape[1] + 1) // 2):
                        for x1 in range((-kernel_shape[0] + 1) // 2, (kernel_shape[0] + 1) // 2):
                            if 0 <= x + x1 <= self.shape[0] - 1 \
                                    and 0 <= y + y1 <= self.shape[1] - 1:
                                if self[x + x1, y + y1]:
                                    blurring_mask[x + x1, y + y1] = False
                            else:
                                raise exc.MaskException(
                                    "setup_blurring_mask extends beyond the sub_grid_size of the mask - pad the image"
                                    "before masking")

        return Mask(blurring_mask, self.pixel_scale)


class SparseMask(Mask):
    def __new__(cls, mask, sparse_grid_size, *args, **kwargs):
        sparse_mask = np.full(mask.shape, True)

        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if not mask[x, y]:
                    if x % sparse_grid_size == 0 and y % sparse_grid_size == 0:
                        sparse_mask[x, y] = False

        return np.array(sparse_mask).view(cls)

    def __init__(self, mask, sparse_grid_size):
        super().__init__(mask)
        self.mask = mask
        self.sparse_grid_size = sparse_grid_size

    @property
    @Memoizer()
    def index_image(self):
        """
        Setup an image which, for each *False* entry in the sparse mask, puts the sparse pixel index in that pixel.

         This is used for computing the image_to_cluster vector, whereby each image pixel is paired to the sparse
         pixel in this image via a neighbor search."""

        sparse_index_2d = np.zeros(self.shape, dtype=int)
        sparse_pixel_index = 0

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if not self[x, y]:
                    sparse_pixel_index += 1
                    sparse_index_2d[x, y] = sparse_pixel_index

        return sparse_index_2d

    @property
    @Memoizer()
    def sparse_to_image(self):
        """
        Compute the mapping of each sparse image pixel to its closest image pixel, defined using a mask of image \
        data_to_pixels.

        Returns
        -------
        cluster_to_image : ndarray
            The mapping between every sparse clustering image pixel and image pixel, where each entry gives the 1D index
            of the image pixel in the self.
        """
        sparse_to_image = np.empty(0, dtype=int)
        image_pixel_index = 0

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):

                if not self[x, y]:
                    sparse_to_image = np.append(sparse_to_image, image_pixel_index)

                if not self.mask[x, y]:
                    image_pixel_index += 1

        return sparse_to_image

    @property
    @Memoizer()
    def image_to_sparse(self):
        """Compute the mapping between every image pixel in the mask and its closest sparse clustering pixel.

        This is performed by going to each image pixel in the *mask*, and pairing it with its nearest neighboring pixel
        in the *sparse_mask*. The index of the *sparse_mask* pixel is drawn from the *sparse_index_image*. This
        neighbor search continue grows larger and larger around a pixel, until a pixel contained in the *sparse_mask* is
        successfully found.

        Returns
        -------
        image_to_cluster : ndarray
            The mapping between every image pixel and its closest sparse clustering pixel, where each entry give the 1D
            index of the sparse pixel in sparse_pixel arrays.

        """
        image_to_sparse = np.empty(0, dtype=int)

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if not self.mask[x, y]:
                    iboarder = 0
                    pixel_match = False
                    while not pixel_match:
                        for x1 in range(x - iboarder, x + iboarder + 1):
                            for y1 in range(y - iboarder, y + iboarder + 1):
                                if 0 <= x1 < self.shape[0] and 0 <= y1 < self.shape[1]:
                                    if not self[x1, y1] and not pixel_match:
                                        image_to_sparse = np.append(image_to_sparse, self.index_image[x1, y1] - 1)
                                        pixel_match = True

                        iboarder += 1
                        if iboarder == 100:
                            raise exc.MaskException('compute_image_to_sparse - Stuck in infinite loop')

        return image_to_sparse


class SubCoordinateGrid(np.ndarray):
    def __new__(cls, mask, sub_grid_size=1, **kwargs):
        sub_pixel_count = 0

        grid = np.zeros(shape=(mask.pixels_in_mask * sub_grid_size ** 2, 2))

        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if not mask[x, y]:
                    x_arcsec, y_arcsec = mask.pixel_coordinates_to_arc_second_coordinates((x, y))

                    for x1 in range(sub_grid_size):
                        for y1 in range(sub_grid_size):
                            grid[sub_pixel_count, 0] = mask.sub_pixel_to_coordinate(x1, x_arcsec, sub_grid_size)

                            grid[sub_pixel_count, 1] = mask.sub_pixel_to_coordinate(y1, y_arcsec, sub_grid_size)

                            sub_pixel_count += 1
        return grid.view(cls)

    def __init__(self, mask, sub_grid_size=1):
        # noinspection PyArgumentList
        super(SubCoordinateGrid, self).__init__()
        self.sub_to_image = mask.sub_to_image_with_size(sub_grid_size)
        self.sub_grid_size = sub_grid_size
        self.sub_grid_length = int(sub_grid_size ** 2.0)
        self.sub_grid_fraction = 1.0 / self.sub_grid_length

    def sub_data_to_image(self, data):
        return np.multiply(self.sub_grid_fraction, data.reshape(-1, self.sub_grid_length).sum(axis=1))
