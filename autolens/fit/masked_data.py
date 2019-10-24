import numpy as np

import autoarray as aa
from autoarray.fit import masked_data
from autolens.fit import positions_fit
from autolens import exc

import copy


class AbstractLensMasked(object):

    def __init__(self, positions, positions_threshold, preload_pixelization_grids_of_planes):

        if positions is not None:
            self.positions = list(
                map(lambda position_set: aa.irregular_grid.manual_1d(grid=np.asarray(position_set)), positions)
            )
        else:
            self.positions = positions

        self.positions_threshold = positions_threshold

        self.preload_pixelization_grids_of_planes = preload_pixelization_grids_of_planes

    def check_positions_trace_within_threshold_via_tracer(self, tracer):

        if (
                self.positions is not None
                and self.positions_threshold is not None
        ):

            traced_positions_of_planes = tracer.traced_positions_of_planes_from_positions(
                positions=self.positions
            )

            fit = positions_fit.LensPositionFit(
                positions=traced_positions_of_planes[-1],
                noise_map=self.imaging.pixel_scales,
            )

            if not fit.maximum_separation_within_threshold(
                    self.positions_threshold
            ):
                raise exc.RayTracingException

    def check_inversion_pixels_are_below_limit_via_tracer(self, tracer):

        if self.inversion_pixel_limit is not None:
            pixelizations = list(filter(None, tracer.pixelizations_of_planes))
            if pixelizations:
                for pixelization in pixelizations:
                    if pixelization.pixels > self.inversion_pixel_limit:
                        raise exc.PixelizationException


class MaskedImaging(masked_data.MaskedImaging, AbstractLensMasked):
    def __init__(
            self,
            imaging,
            mask,
            trimmed_psf_shape_2d=None,
            pixel_scale_interpolation_grid=None,
            inversion_pixel_limit=None,
            inversion_uses_border=True,
            hyper_noise_map_max=None,
            positions=None,
            positions_threshold=None,
            preload_pixelization_grids_of_planes=None,
    ):
        """
        The lens data is the collection of data_type (image, noise-map, PSF), a mask, grid, convolver \
        and other utilities that are used for modeling and fitting an image of a strong lens.

        Whilst the image, noise-map, etc. are loaded in 2D, the lens data creates reduced 1D arrays of each \
        for lensing calculations.

        Parameters
        ----------
        imaging: im.Imaging
            The imaging data_type all in 2D (the image, noise-map, PSF, etc.)
        mask: msk.Mask
            The 2D mask that is applied to the image.
        sub_size : int
            The size of the sub-grid used for each lens SubGrid. E.g. a value of 2 grid each image-pixel on a 2x2 \
            sub-grid.
        trimmed_psf_shape_2d : (int, int)
            The shape of the PSF used for convolving model image generated using analytic light profiles. A smaller \
            shape will trim the PSF relative to the input image PSF, giving a faster analysis run-time.
        positions : [[]]
            Lists of image-pixel coordinates (arc-seconds) that mappers close to one another in the source-plane(s), \
            used to speed up the non-linear sampling.
        pixel_scale_interpolation_grid : float
            If *True*, expensive to compute mass profile deflection angles will be computed on a sparse grid and \
            interpolated to the grid, sub and blurring grids.
        inversion_pixel_limit : int or None
            The maximum number of pixels that can be used by an inversion, with the limit placed primarily to speed \
            up run.
        """

        super(MaskedImaging, self).__init__(
            imaging=imaging,
            mask=mask,
            trimmed_psf_shape_2d=trimmed_psf_shape_2d,
            pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
            inversion_pixel_limit=inversion_pixel_limit,
            inversion_uses_border=inversion_uses_border,
            hyper_noise_map_max=hyper_noise_map_max,
        )

        AbstractLensMasked.__init__(self=self, positions=positions,
            positions_threshold=positions_threshold, preload_pixelization_grids_of_planes=preload_pixelization_grids_of_planes)

    def binned_from_bin_up_factor(self, bin_up_factor):

        binned_imaging = self.imaging.binned_from_bin_up_factor(
            bin_up_factor=bin_up_factor
        )
        binned_mask = self.mask.mapping.binned_mask_from_bin_up_factor(bin_up_factor=bin_up_factor)

        return self.__class__(
            imaging=binned_imaging,
            mask=binned_mask,
            trimmed_psf_shape_2d=self.trimmed_psf_shape_2d,
            pixel_scale_interpolation_grid=self.pixel_scale_interpolation_grid,
            inversion_pixel_limit=self.inversion_pixel_limit,
            inversion_uses_border=self.inversion_uses_border,
            hyper_noise_map_max=self.hyper_noise_map_max,
            positions=self.positions,
            positions_threshold=self.positions_threshold,
            preload_pixelization_grids_of_planes=self.preload_pixelization_grids_of_planes
        )

    def signal_to_noise_limited_from_signal_to_noise_limit(self, signal_to_noise_limit):

        imaging_with_signal_to_noise_limit = self.imaging.signal_to_noise_limited_from_signal_to_noise_limit(
            signal_to_noise_limit=signal_to_noise_limit
        )

        return self.__class__(
            imaging=imaging_with_signal_to_noise_limit,
            mask=self.mask,
            trimmed_psf_shape_2d=self.trimmed_psf_shape_2d,
            pixel_scale_interpolation_grid=self.pixel_scale_interpolation_grid,
            inversion_pixel_limit=self.inversion_pixel_limit,
            inversion_uses_border=self.inversion_uses_border,
            hyper_noise_map_max=self.hyper_noise_map_max,
            positions=self.positions,
            positions_threshold=self.positions_threshold,
            preload_pixelization_grids_of_planes=self.preload_pixelization_grids_of_planes
        )


class MaskedInterferometer(masked_data.MaskedInterferometer):
    def __init__(
            self,
            interferometer,
            mask,
            trimmed_primary_beam_shape_2d=None,
            pixel_scale_interpolation_grid=None,
            inversion_pixel_limit=None,
            inversion_uses_border=True,
            hyper_noise_map_max=None,
            positions=None,
            positions_threshold=None,
            preload_pixelization_grids_of_planes=None,
    ):
        """
        The lens data is the collection of data_type (image, noise-map, primary_beam), a mask, grid, convolver \
        and other utilities that are used for modeling and fitting an image of a strong lens.

        Whilst the image, noise-map, etc. are loaded in 2D, the lens data creates reduced 1D arrays of each \
        for lensing calculations.

        Parameters
        ----------
        imaging: im.Imaging
            The imaging data_type all in 2D (the image, noise-map, primary_beam, etc.)
        mask: msk.Mask
            The 2D mask that is applied to the image.
        sub_size : int
            The size of the sub-grid used for each lens SubGrid. E.g. a value of 2 grid each image-pixel on a 2x2 \
            sub-grid.
        trimmed_primary_beam_shape_2d : (int, int)
            The shape of the primary_beam used for convolving model image generated using analytic light profiles. A smaller \
            shape will trim the primary_beam relative to the input image primary_beam, giving a faster analysis run-time.
        positions : [[]]
            Lists of image-pixel coordinates (arc-seconds) that mappers close to one another in the source-plane(s), \
            used to speed up the non-linear sampling.
        pixel_scale_interpolation_grid : float
            If *True*, expensive to compute mass profile deflection angles will be computed on a sparse grid and \
            interpolated to the grid, sub and blurring grids.
        inversion_pixel_limit : int or None
            The maximum number of pixels that can be used by an inversion, with the limit placed primarily to speed \
            up run.
        """

        self.interferometer = interferometer

        super(MaskedInterferometer, self).__init__(
            interferometer=interferometer,
            mask=mask,
            trimmed_primary_beam_shape_2d=trimmed_primary_beam_shape_2d,
            pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
            inversion_pixel_limit=inversion_pixel_limit,
            inversion_uses_border=inversion_uses_border,
            hyper_noise_map_max=hyper_noise_map_max,
        )

        AbstractLensMasked.__init__(self=self, positions=positions,
            positions_threshold=positions_threshold, preload_pixelization_grids_of_planes=preload_pixelization_grids_of_planes)


