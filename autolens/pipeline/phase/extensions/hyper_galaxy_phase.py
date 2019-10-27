import copy

import numpy as np
from typing import cast

import autoarray as aa
import autofit as af
from autolens.fit import fit
from autolens.fit import masked_data
from autoastro.galaxy import galaxy as g
from autoastro.hyper import hyper_data as hd
from autolens.pipeline.phase import imaging
from autolens.pipeline import visualizer
from .hyper_phase import HyperPhase


class Analysis(af.Analysis):
    def __init__(
            self, masked_imaging, hyper_model_image, hyper_galaxy_image, image_path
    ):
        """
        An analysis to fit the noise for a single galaxy image.
        Parameters
        ----------
        masked_imaging: LensData
            lens simulate, including an image and noise
        hyper_model_image: ndarray
            An image produce of the overall system by a model
        hyper_galaxy_image: ndarray
            The contribution of one galaxy to the model image
        """

        self.masked_imaging = masked_imaging

        self.visualizer = visualizer.HyperGalaxyVisualizer(image_path)

        self.hyper_model_image = hyper_model_image
        self.hyper_galaxy_image = hyper_galaxy_image

    def visualize(self, instance, during_analysis):

        if self.visualizer.plot_hyper_galaxy_subplot:
            hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)

            hyper_background_noise = self.hyper_background_noise_for_instance(
                instance=instance
            )

            contribution_map = instance.hyper_galaxy.contribution_map_from_hyper_images(
                hyper_model_image=self.hyper_model_image,
                hyper_galaxy_image=self.hyper_galaxy_image,
            )

            fit_normal = aa.fit_imaging(
                image=self.masked_imaging.image,
                noise_map=self.masked_imaging.noise_map,
                mask=self.masked_imaging.mask,
                model_image=self.hyper_model_image,
            )

            fit_hyper = self.fit_for_hyper_galaxy(
                hyper_galaxy=instance.hyper_galaxy,
                hyper_image_sky=hyper_image_sky,
                hyper_background_noise=hyper_background_noise,
            )

            self.visualizer.hyper_galaxy_subplot(
                hyper_galaxy_image=self.hyper_galaxy_image,
                contribution_map=contribution_map,
                noise_map=self.masked_imaging.noise_map,
                hyper_noise_map=fit_hyper.noise_map,
                chi_squared_map=fit_normal.chi_squared_map,
                hyper_chi_squared_map=fit_hyper.chi_squared_map,
            )

    def fit(self, instance):
        """
        Fit the model image to the real image by scaling the hyper_galaxies noise.
        Parameters
        ----------
        instance: ModelInstance
            A model instance with a hyper_galaxies galaxy property
        Returns
        -------
        fit: float
        """

        hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)

        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        fit = self.fit_for_hyper_galaxy(
            hyper_galaxy=instance.hyper_galaxy,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

        return fit.figure_of_merit

    @staticmethod
    def hyper_image_sky_for_instance(instance):
        if hasattr(instance, "hyper_image_sky"):
            return instance.hyper_image_sky

    @staticmethod
    def hyper_background_noise_for_instance(instance):
        if hasattr(instance, "hyper_background_noise"):
            return instance.hyper_background_noise

    def fit_for_hyper_galaxy(
            self, hyper_galaxy, hyper_image_sky, hyper_background_noise
    ):

        image = fit.hyper_image_from_image_and_hyper_image_sky(
            image=self.masked_imaging.image, hyper_image_sky=hyper_image_sky
        )
        
        if hyper_background_noise is not None:
            noise_map = hyper_background_noise.hyper_noise_map_from_noise_map(
                noise_map=self.masked_imaging.noise_map
            )
        else:
            noise_map = self.masked_imaging.noise_map

        hyper_noise_map = hyper_galaxy.hyper_noise_map_from_hyper_images_and_noise_map(
            hyper_model_image=self.hyper_model_image,
            hyper_galaxy_image=self.hyper_galaxy_image,
            noise_map=self.masked_imaging.noise_map,
        )

        noise_map = noise_map + hyper_noise_map

        return aa.fit_imaging(
            image=image,
            noise_map=noise_map,
            mask=self.masked_imaging.mask,
            model_image=self.hyper_model_image,
        )

    @classmethod
    def describe(cls, instance):
        return "Running hyper_galaxies galaxy fit for HyperGalaxy:\n{}".format(
            instance.hyper_galaxy
        )


class HyperGalaxyPhase(HyperPhase):
    Analysis = Analysis

    def __init__(self, phase):

        super().__init__(phase=phase, hyper_name="hyper_galaxy")
        self.include_sky_background = False
        self.include_noise_background = False

    def run_hyper(self, data, results=None):
        """
        Run a fit for each galaxy from the previous phase.
        Parameters
        ----------
        data: LensData
        results: ResultsCollection
            Results from all previous phases
        Returns
        -------
        results: HyperGalaxyResults
            A collection of results, with one item per a galaxy
        """
        phase = self.make_hyper_phase()

        masked_imaging = masked_data.MaskedImaging(
            imaging=data,
            mask=results.last.mask,
            trimmed_psf_shape_2d=cast(imaging.PhaseImaging, phase).meta_data_fit.trimmed_psf_shape_2d,
            positions=results.last.positions,
            positions_threshomasked_imaging=cast(
                imaging.PhaseImaging, phase
            ).meta_data_fit.positions_threshold,
            pixel_scale_interpolation_grid=cast(
                imaging.PhaseImaging, phase
            ).meta_data_fit.pixel_scale_interpolation_grid,
            inversion_pixel_limit=cast(
                imaging.PhaseImaging, phase
            ).meta_data_fit.inversion_pixel_limit,
            inversion_uses_border=cast(
                imaging.PhaseImaging, phase
            ).meta_data_fit.inversion_uses_border,
            preload_pixelization_grids_of_planes=None,
        )

        hyper_result = copy.deepcopy(results.last)
        hyper_result.variable = hyper_result.variable.copy_with_fixed_priors(
            hyper_result.constant
        )

        hyper_result.analysis.hyper_model_image = results.last.hyper_model_image
        hyper_result.analysis.hyper_galaxy_image_path_dict = (
            results.last.hyper_galaxy_image_path_dict
        )

        for path, galaxy in results.last.path_galaxy_tuples:

            optimizer = phase.optimizer.copy_with_name_extension(extension=path[-1])

            optimizer.paths.phase_tag = ""

            # TODO : This is a HACK :O

            optimizer.variable.galaxies = []

            optimizer.const_efficiency_mode = af.conf.instance.non_linear.get(
                "MultiNest", "extension_hyper_galaxy_const_efficiency_mode", bool
            )
            optimizer.sampling_efficiency = af.conf.instance.non_linear.get(
                "MultiNest", "extension_hyper_galaxy_sampling_efficiency", float
            )
            optimizer.n_live_points = af.conf.instance.non_linear.get(
                "MultiNest", "extension_hyper_galaxy_n_live_points", int
            )
            optimizer.multimodal = af.conf.instance.non_linear.get(
                "MultiNest", "extension_hyper_galaxy_multimodal", bool
            )

            optimizer.variable.hyper_galaxy = g.HyperGalaxy

            if self.include_sky_background:
                optimizer.variable.hyper_image_sky = hd.HyperImageSky

            if self.include_noise_background:
                optimizer.variable.hyper_background_noise = hd.HyperBackgroundNoise

            # If array is all zeros, galaxy did not have image in previous phase and
            # shoumasked_imaging be ignored
            if not np.all(hyper_result.analysis.hyper_galaxy_image_path_dict[path] == 0):

                analysis = self.Analysis(
                    masked_imaging=masked_imaging,
                    hyper_model_image=hyper_result.analysis.hyper_model_image,
                    hyper_galaxy_image=hyper_result.analysis.hyper_galaxy_image_path_dict[path],
                    image_path=optimizer.paths.image_path,
                )

                result = optimizer.fit(analysis)

                def transfer_field(name):
                    if hasattr(result.constant, name):
                        setattr(
                            hyper_result.constant.object_for_path(path),
                            name,
                            getattr(result.constant, name),
                        )
                        setattr(
                            hyper_result.variable.object_for_path(path),
                            name,
                            getattr(result.variable, name),
                        )

                transfer_field("hyper_galaxy")

                hyper_result.constant.hyper_image_sky = getattr(
                    result.constant, "hyper_image_sky"
                )
                hyper_result.variable.hyper_image_sky = getattr(
                    result.variable, "hyper_image_sky"
                )

                hyper_result.constant.hyper_background_noise = getattr(
                    result.constant, "hyper_background_noise"
                )
                hyper_result.variable.hyper_background_noise = getattr(
                    result.variable, "hyper_background_noise"
                )

        return hyper_result


class HyperGalaxyBackgroundSkyPhase(HyperGalaxyPhase):
    def __init__(self, phase):
        super().__init__(phase=phase)
        self.include_sky_background = True
        self.include_noise_background = False


class HyperGalaxyBackgroundNoisePhase(HyperGalaxyPhase):
    def __init__(self, phase):
        super().__init__(phase=phase)
        self.include_sky_background = False
        self.include_noise_background = True


class HyperGalaxyBackgroundBothPhase(HyperGalaxyPhase):
    def __init__(self, phase):
        super().__init__(phase=phase)
        self.include_sky_background = True
        self.include_noise_background = True
