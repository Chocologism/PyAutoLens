import autofit as af
import autoarray as aa
from autolens import exc
from autoarray.operators.inversion import pixelizations as pix
from autolens.pipeline.phase.data.phase import default_mask_function, isinstance_or_prior


class MetaDataFit:
    def __init__(
            self,
            variable,
            sub_size=2,
            signal_to_noise_limit=None,
            positions_threshold=None,
            mask_function=None,
            inner_mask_radii=None,
            pixel_scale_interpolation_grid=None,
            inversion_uses_border=True,
            inversion_pixel_limit=None,
            is_hyper_phase=False
    ):
        self.is_hyper_phase = is_hyper_phase
        self.variable = variable
        self.sub_size = sub_size
        self.signal_to_noise_limit = signal_to_noise_limit
        self.positions_threshold = positions_threshold
        self.mask_function = mask_function
        self.inner_mask_radii = inner_mask_radii
        self.pixel_scale_interpolation_grid = pixel_scale_interpolation_grid
        self.inversion_uses_border = inversion_uses_border
        self.inversion_pixel_limit = (
                inversion_pixel_limit or
                af.conf.instance.general.get(
                    "inversion",
                    "inversion_pixel_limit_overall",
                    int
                )
        )

    def setup_phase_mask(self, data, mask):

        if self.mask_function is not None:
            mask = self.mask_function(image=data.image, sub_size=self.sub_size)
        elif mask is None and self.mask_function is None:
            mask = default_mask_function(image=data.image)

        if mask.sub_size != self.sub_size:
            mask = mask.new_mask_with_new_sub_size(sub_size=self.sub_size)

        if self.inner_mask_radii is not None:
            inner_mask = aa.mask.circular(
                shape=mask.shape,
                pixel_scales=mask.pixel_scales,
                radius_arcsec=self.inner_mask_radii,
                sub_size=self.sub_size,
                invert=True,
            )
            mask = mask + inner_mask

        return mask

    def check_positions(self, positions):

        if self.positions_threshold is not None and positions is None:
            raise exc.PhaseException(
                "You have specified for a phase to use positions, but not input positions to the "
                "pipeline when you ran it."
            )

    @property
    def pixelization(self):
        for galaxy in self.variable.galaxies:
            if galaxy.pixelization is not None:
                if isinstance(galaxy.pixelization, af.PriorModel):
                    return galaxy.pixelization.cls
                else:
                    return galaxy.pixelization

    @property
    def uses_cluster_inversion(self):
        if self.variable.galaxies:
            for galaxy in self.variable.galaxies:
                if isinstance_or_prior(galaxy.pixelization, pix.VoronoiBrightnessImage):
                    return True
        return False

    def preload_pixelization_grids_of_planes_from_results(self, results):

        if self.is_hyper_phase:
            return None

        if (
                results is not None
                and results.last is not None
                and hasattr(results.last, "hyper_combined")
                and self.pixelization is not None
        ):
            if self.pixelization.__class__ is results.last.pixelization.__class__:
                return (
                    results.last.hyper_combined.most_likely_pixelization_grids_of_planes
                )
        return None