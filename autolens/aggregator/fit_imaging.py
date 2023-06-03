from typing import Optional, List

import autofit as af
import autoarray as aa
import autogalaxy as ag

from autogalaxy.aggregator.imaging import _imaging_from
from autogalaxy.aggregator.abstract import AbstractAgg

from autolens.imaging.fit_imaging import FitImaging
from autolens.analysis.preloads import Preloads
from autolens.aggregator.tracer import _tracer_from


def _fit_imaging_from(
    fit: af.Fit,
    galaxies: List[ag.Galaxy],
    settings_dataset: aa.SettingsImaging = None,
    settings_pixelization: aa.SettingsPixelization = None,
    settings_inversion: aa.SettingsInversion = None,
    use_preloaded_grid: bool = True,
) -> FitImaging:
    """
    Returns a `FitImaging` object from a PyAutoFit database `Fit` object and an instance of galaxies from a non-linear
    search model-fit.

    This function adds the `adapt_model_image` and `adapt_galaxy_image_path_dict` to the galaxies before performing the
    fit, if they were used.

    Parameters
    ----------
    fit
        A PyAutoFit database Fit object containing the generators of the results of model-fits.
    galaxies
        A list of galaxies corresponding to a sample of a non-linear search and model-fit.

    Returns
    -------
    FitImaging
        The fit to the imaging dataset computed via an instance of galaxies.
    """

    dataset = _imaging_from(fit=fit, settings_dataset=settings_dataset)

    tracer = _tracer_from(fit=fit, galaxies=galaxies)

    settings_pixelization = settings_pixelization or fit.value(
        name="settings_pixelization"
    )
    settings_inversion = settings_inversion or fit.value(name="settings_inversion")

    preloads = Preloads(use_w_tilde=False)

    if use_preloaded_grid:
        sparse_grids_of_planes = fit.value(name="preload_sparse_grids_of_planes")

        if sparse_grids_of_planes is not None:
            preloads = Preloads(
                sparse_image_plane_grid_pg_list=sparse_grids_of_planes,
                use_w_tilde=False,
            )

            if len(preloads.sparse_image_plane_grid_pg_list) == 2:
                if type(preloads.sparse_image_plane_grid_pg_list[1]) != list:
                    preloads.sparse_image_plane_grid_pg_list[1] = [
                        preloads.sparse_image_plane_grid_pg_list[1]
                    ]

    return FitImaging(
        dataset=dataset,
        tracer=tracer,
        settings_pixelization=settings_pixelization,
        settings_inversion=settings_inversion,
        preloads=preloads,
    )


class FitImagingAgg(AbstractAgg):
    def __init__(
        self,
        aggregator: af.Aggregator,
        settings_dataset: Optional[aa.SettingsImaging] = None,
        settings_pixelization: Optional[aa.SettingsPixelization] = None,
        settings_inversion: Optional[aa.SettingsInversion] = None,
        use_preloaded_grid: bool = True,
    ):
        """
        Wraps a PyAutoFit aggregator in order to create generators of fits to imaging data, corresponding to the
        results of a non-linear search model-fit.
        """
        super().__init__(aggregator=aggregator)

        self.settings_dataset = settings_dataset
        self.settings_pixelization = settings_pixelization
        self.settings_inversion = settings_inversion
        self.use_preloaded_grid = use_preloaded_grid

    def object_via_gen_from(self, fit, galaxies) -> FitImaging:
        """
        Creates a `FitImaging` object from a `ModelInstance` that contains the galaxies of a sample from a non-linear
        search.

        Parameters
        ----------
        fit
            A PyAutoFit database Fit object containing the generators of the results of model-fits.
        galaxies
            A list of galaxies corresponding to a sample of a non-linear search and model-fit.

        Returns
        -------
        FitImaging
            A fit to imaging data whose galaxies are a sample of a PyAutoFit non-linear search.
        """

        return _fit_imaging_from(
            fit=fit,
            galaxies=galaxies,
            settings_dataset=self.settings_dataset,
            settings_pixelization=self.settings_pixelization,
            settings_inversion=self.settings_inversion,
            use_preloaded_grid=self.use_preloaded_grid,
        )
