import numpy as np
from autolens.imaging import masked_image as mi
from autolens.imaging import mask
from autolens.analysis import ray_tracing
from autolens import exc

# TODO : Can we make hyper_model_image, image_plane_lens_galaxy_images, minimum_Values a part of hyper galaxies?

minimum_value_profile = 0.1


class AbstractFitter(object):

    def __init__(self, masked_image):
        self.masked_image = masked_image

    @property
    def noise_term(self):
        return noise_term_from_noise(self.masked_image.noise)


class AbstractHyperFitter(AbstractFitter):

    def __init__(self, masked_image, tracer=None, hyper_model_image=None, hyper_galaxy_images=None,
                 hyper_minimum_values=None):
        super().__init__(masked_image)
        self.tracer = tracer
        self.hyper_model_image = hyper_model_image
        self.hyper_galaxy_images = hyper_galaxy_images
        self.hyper_minimum_values = hyper_minimum_values

    @property
    def contributions(self):
        return contributions_from_hyper_images_and_galaxies(self.hyper_model_image, self.hyper_galaxy_images,
                                                            self.tracer.hyper_galaxies, self.hyper_minimum_values)

    @property
    def scaled_noise(self):

        return scaled_noise_from_hyper_galaxies_and_contributions(self.contributions, self.tracer.hyper_galaxies,
                                                                  self.masked_image.noise)

    @property
    def scaled_noise_term(self):
        return noise_term_from_noise(self.scaled_noise)

    @property
    def scaled_noise_2d(self):
        return self.masked_image.map_to_2d(self.scaled_noise)


class ProfileFitter(AbstractFitter):

    def __init__(self, masked_image, tracer):
        """
        Class to evaluate the fit between a model described by a tracer and an actual masked_image.

        Parameters
        ----------
        masked_image: mi.MaskedImage
            An masked_image that has been masked for efficiency
        tracer: ray_tracing.TracerImageSourcePlanes
            An object describing the model
        """
        super(ProfileFitter, self).__init__(masked_image)
        self.tracer = tracer
        self.blurred_image_plane_image = self.masked_image.convolver_image.convolve_image(self.image_plane_image, 
                                                                                          self.image_plane_blurring_image)

    @property
    def image_plane_image(self):
        return self.tracer.image_plane_image

    @property
    def image_plane_blurring_image(self):
        return self.tracer.image_plane_blurring_image

    @property
    def blurred_image_plane_image_residuals(self):
        return residuals_from_image_and_model(self.masked_image, self.blurred_image_plane_image)

    @property
    def blurred_image_plane_image_chi_squareds(self):
        return chi_squareds_from_residuals_and_noise(self.blurred_image_plane_image_residuals, self.masked_image.noise)

    @property
    def blurred_image_plane_image_chi_squared_term(self):
        return chi_squared_term_from_chi_squareds(self.blurred_image_plane_image_chi_squareds)

    @property
    def blurred_image_plane_image_likelihood(self):
        """
        Fit the data_vector using the ray_tracing model, where only light_profiles are used to represent the galaxy
        images.
        """
        return likelihood_from_chi_squared_and_noise_terms(self.blurred_image_plane_image_chi_squared_term, self.noise_term)

    def pixelization_fitter_with_profile_subtracted_masked_image(self, sparse_mask):
        return PixelizationFitter(self.masked_image[:] - self.blurred_image_plane_image, sparse_mask, self.tracer)

    @property
    def blurred_image_plane_images_of_planes(self):
        return list(map(lambda image_plane_image, image_plane_blurring_image :
                        self.masked_image.convolver_image.convolve_image(image_plane_image, image_plane_blurring_image),
                        self.tracer.image_plane_images_of_planes, self.tracer.image_plane_blurring_images_of_planes))

    @property
    def blurred_image_plane_images_of_galaxies(self):
        return list(map(lambda image_plane_image, image_plane_blurring_image :
                        self.masked_image.convolver_image.convolve_image(image_plane_image, image_plane_blurring_image),
                        self.tracer.image_plane_images_of_galaxies, self.tracer.image_plane_blurring_images_of_galaxies))

    @property
    def blurred_image_plane_image_2d(self):
        return self.masked_image.map_to_2d(self.blurred_image_plane_image)

    @property
    def blurred_image_plane_image_residuals_2d(self):
        return self.masked_image.map_to_2d(self.blurred_image_plane_image_residuals)

    @property
    def blurred_image_plane_image_chi_squareds_2d(self):
        return self.masked_image.map_to_2d(self.blurred_image_plane_image_chi_squareds)

    @property
    def blurred_image_plane_images_of_planes_2d(self):
        return list(map(lambda blurred_plane_image_plane_image:
                        self.masked_image.map_to_2d(blurred_plane_image_plane_image),
                        self.blurred_image_plane_images_of_planes))

    @property
    def blurred_image_plane_images_of_galaxies_2d(self):
        return list(map(lambda galaxy_image_plane_image: self.masked_image.map_to_2d(galaxy_image_plane_image),
                        self.blurred_image_plane_images_of_galaxies))

    def plane_images_of_planes_2d(self, shape=(30, 30)):

        def map_to_2d(image, shape):

            image_2d = np.zeros(shape)

            for x in range(shape[0]):
                for y in range(shape[1]):

                    image_2d[y, x] = image[(x)*shape[0] + y]

            return image_2d

        return list(map(lambda plane_image : map_to_2d(plane_image, shape), self.tracer.plane_images_of_planes(shape)))


class HyperProfileFitter(ProfileFitter, AbstractHyperFitter):

    def __init__(self, masked_image, tracer, hyper_model_image, hyper_galaxy_images, hyper_minimum_values):
        """
        Class to evaluate the fit between a model described by a tracer and an actual masked_image.

        Parameters
        ----------
        masked_image: mi.MaskedImage
            An masked_image that has been masked for efficiency
        tracer: ray_tracing.TracerImageSourcePlanes
            An object describing the model
        """
        super(HyperProfileFitter, self).__init__(masked_image, tracer)
        AbstractHyperFitter.__init__(self, masked_image, tracer, hyper_model_image, hyper_galaxy_images,
                                     hyper_minimum_values)

    @property
    def blurred_image_plane_image_scaled_chi_squareds(self):
        return chi_squareds_from_residuals_and_noise(self.blurred_image_plane_image_residuals, self.scaled_noise)

    @property
    def blurred_image_plane_image_scaled_chi_squared_term(self):
        return chi_squared_term_from_chi_squareds(self.blurred_image_plane_image_scaled_chi_squareds)

    @property
    def blurred_image_plane_image_scaled_likelihood(self):
        """
        Fit the data_vector using the ray_tracing model, where only light_profiles are used to represent the galaxy
        images.
        """
        return likelihood_from_chi_squared_and_noise_terms(self.blurred_image_plane_image_scaled_chi_squared_term,
                                                           self.scaled_noise_term)

    @property
    def blurred_image_plane_image_scaled_chi_squareds_2d(self):
        return self.masked_image.map_to_2d(self.blurred_image_plane_image_scaled_chi_squareds)

    def pixelization_fitter_with_profile_subtracted_masked_image(self, sparse_mask):
        return HyperPixelizationFitter(self.masked_image[:] - self.blurred_image_plane_image, sparse_mask, self.tracer,
                                       self.hyper_model_image, self.hyper_galaxy_images, self.hyper_minimum_values)


class PixelizationFitter(AbstractFitter):

    def __init__(self, masked_image, sparse_mask, tracer, perform_reconstruction=True):
        """
        Class to evaluate the fit between a model described by a tracer and an actual masked_image.

        Parameters
        ----------
        masked_image: mi.MaskedImage
            An masked_image that has been masked for efficiency
        sparse_mask: mask.SparseMask | None
            A mask describing which pixels should be used in clustering for pixelizations
        tracer: ray_tracing.TracerImageSourcePlanes
            An object describing the model
        """
        super().__init__(masked_image)
        self.masked_image = masked_image
        self.tracer = tracer
        self.sparse_mask = sparse_mask

        # TODO : This if loop is required to stop the HyperPixelizationFitter waste time fitting the data with the
        # TODO : unscaled noise during inheritance. Prob a better way to handle this.

        if perform_reconstruction:
            self.reconstruction = self.reconstructors.reconstruction_from_reconstructor_and_data(
                self.masked_image,
                self.masked_image.noise,
                self.masked_image.convolver_mapping_matrix)

    @property
    def reconstructors(self):
        return self.tracer.reconstructors_from_source_plane(self.masked_image.borders, self.sparse_mask)

    @property
    def reconstructed_image_plane_image(self):
        return self.reconstruction.reconstructed_image

    @property
    def reconstructed_image_plane_image_residuals(self):
        return residuals_from_image_and_model(self.masked_image, self.reconstruction.reconstructed_image)

    @property
    def reconstructed_image_plane_image_chi_squareds(self):
        return chi_squareds_from_residuals_and_noise(self.reconstructed_image_plane_image_residuals, self.masked_image.noise)

    @property
    def reconstructed_image_plane_image_chi_squared_term(self):
        return chi_squared_term_from_chi_squareds(self.reconstructed_image_plane_image_chi_squareds)

    @property
    def reconstructed_image_plane_image_evidence(self):
        return evidence_from_reconstruction_terms(self.reconstructed_image_plane_image_chi_squared_term,
                                                  self.reconstruction.regularization_term,
                                                  self.reconstruction.log_det_curvature_reg_matrix_term,
                                                  self.reconstruction.log_det_regularization_matrix_term,
                                                  self.noise_term)

    @property
    def reconstructed_image_plane_image_2d(self):
        return self.masked_image.map_to_2d(self.reconstructed_image_plane_image)

    @property
    def reconstructed_image_plane_image_residuals_2d(self):
        return self.masked_image.map_to_2d(self.reconstructed_image_plane_image_residuals)

    @property
    def reconstructed_image_plane_image_chi_squareds_2d(self):
        return self.masked_image.map_to_2d(self.reconstructed_image_plane_image_chi_squareds)


class HyperPixelizationFitter(PixelizationFitter, AbstractHyperFitter):

    def __init__(self, masked_image, sparse_mask, tracer, hyper_model_image, hyper_galaxy_images, hyper_minimum_values):
        """
        Class to evaluate the fit between a model described by a tracer and an actual masked_image.

        Parameters
        ----------
        masked_image: mi.MaskedImage
            An masked_image that has been masked for efficiency
        sparse_mask: mask.SparseMask
            A mask describing which pixels should be used in clustering for pixelizations
        tracer: ray_tracing.TracerImageSourcePlanes
            An object describing the model
        """
        super(HyperPixelizationFitter, self).__init__(masked_image, sparse_mask, tracer, False)
        self.hyper_model_image = hyper_model_image
        self.hyper_galaxy_images = hyper_galaxy_images
        self.hyper_minimum_values = hyper_minimum_values
        self._scaled_noise = self.scaled_noise
        self._scaled_noise_term = noise_term_from_noise(self._scaled_noise)
        self.reconstruction = self.reconstructors.reconstruction_from_reconstructor_and_data(
            self.masked_image,
            self._scaled_noise,
            self.masked_image.convolver_mapping_matrix)

    @property
    def reconstructed_image_plane_image_scaled_chi_squareds(self):
        return chi_squareds_from_residuals_and_noise(self.reconstructed_image_plane_image_residuals, self._scaled_noise)

    @property
    def reconstructed_image_plane_image_scaled_chi_squared_term(self):
        return chi_squared_term_from_chi_squareds(self.reconstructed_image_plane_image_scaled_chi_squareds)

    @property
    def reconstructed_image_plane_image_scaled_evidence(self):
        return evidence_from_reconstruction_terms(self.reconstructed_image_plane_image_scaled_chi_squared_term,
                                                  self.reconstruction.regularization_term,
                                                  self.reconstruction.log_det_curvature_reg_matrix_term,
                                                  self.reconstruction.log_det_regularization_matrix_term,
                                                  self._scaled_noise_term)

    @property
    def reconstructed_image_plane_image_scaled_chi_squareds_2d(self):
        return self.masked_image.map_to_2d(self.reconstructed_image_plane_image_scaled_chi_squareds)


def blur_image_including_blurring_region(image, blurring_image, convolver):
    """For a given masked_image and blurring region, convert them to 2D and blur with the PSF, then return as
    the 1D DataGrid.

    Parameters
    ----------
    image : ndarray
        The masked_image data_vector using the GridData 1D representation.
    blurring_image : ndarray
        The blurring region data_vector, using the GridData 1D representation.
    convolver : auto_lens.pixelization.frame_convolution.KernelConvolver
        The 2D Point Spread Function (PSF).
    """
    return convolver.convolve_image(image, blurring_image)


def residuals_from_image_and_model(image, model):
    """Compute the residuals between an observed charge injection masked_image and post-cti model masked_image.

    Residuals = (Data - Model).

    Parameters
    -----------
    image : ChInj.CIImage
        The observed charge injection masked_image data.
    model : np.ndarray
        The model masked_image.
    """
    return np.subtract(image, model)


def chi_squareds_from_residuals_and_noise(residuals, noise):
    """Computes a chi-squared masked_image, by calculating the squared residuals between an observed charge injection \
    images and post-cti hyper_model_image masked_image and dividing by the variance (noises**2.0) in each pixel.

    Chi_Sq = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)

    This gives the residuals, which are divided by the variance of each pixel and squared to give their chi sq.

    Parameters
    -----------
    residuals
    noise : np.ndarray
        The noises in the masked_image.
    """
    return np.square((np.divide(residuals, noise)))


def chi_squared_term_from_chi_squareds(chi_squareds):
    """Compute the chi-squared of a model masked_image's fit to the data_vector, by taking the difference between the
    observed masked_image and model ray-tracing masked_image, dividing by the noise in each pixel and squaring:

    [Chi_Squared] = sum(([Data - Model] / [Noise]) ** 2.0)

    Parameters
    ----------
    chi_squareds
    """
    return np.sum(chi_squareds)


def noise_term_from_noise(noise):
    """Compute the noise normalization term of an masked_image, which is computed by summing the noise in every pixel:

    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

    Parameters
    ----------
    noise : grids.GridData
        The noise in each pixel.
    """
    return np.sum(np.log(2 * np.pi * noise ** 2.0))


def likelihood_from_chi_squared_and_noise_terms(chi_squared_term, noise_term):
    """Compute the likelihood of a model masked_image's fit to the data_vector, by taking the difference between the
    observed masked_image and model ray-tracing masked_image. The likelihood consists of two terms:

    Chi-squared term - The residuals (model - data_vector) of every pixel divided by the noise in each pixel, all
    squared.
    [Chi_Squared_Term] = sum(([Residuals] / [Noise]) ** 2.0)

    The overall normalization of the noise is also included, by summing the log noise value in each pixel:
    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

    These are summed and multiplied by -0.5 to give the likelihood:

    Likelihood = -0.5*[Chi_Squared_Term + Noise_Term]

    Parameters
    ----------
    """
    return -0.5 * (chi_squared_term + noise_term)


def contributions_from_hyper_images_and_galaxies(hyper_model_image, hyper_galaxy_images, hyper_galaxies,
                                                 minimum_values):
    """Use the model masked_image and galaxy masked_image (computed in the previous phase of the pipeline) to determine the
    contributions of each hyper galaxy.

    Parameters
    -----------
    minimum_values
    hyper_model_image : ndarray
        The best-fit model masked_image to the data_vector, from a previous phase of the pipeline
    hyper_galaxy_images : [ndarray]
        The best-fit model masked_image of each hyper-galaxy, which can tell us how much flux each pixel contributes to.
    hyper_galaxies : [galaxy.HyperGalaxy]
        Each hyper-galaxy which is used to determine its contributions.
    """
    # noinspection PyArgumentList
    return list(map(lambda hyper, galaxy_image, minimum_value:
                    hyper.contributions_from_preload_images(hyper_model_image, galaxy_image, minimum_value),
                    hyper_galaxies, hyper_galaxy_images, minimum_values))


def scaled_noise_from_hyper_galaxies_and_contributions(contributions, hyper_galaxies, noise):
    """Use the contributions of each hyper galaxy to compute the scaled noise.
    Parameters
    -----------
    noise
    hyper_galaxies
    contributions : [ndarray]
        The contribution of flux of each galaxy in each pixel (computed from galaxy.HyperGalaxy)
    """
    scaled_noises = list(map(lambda hyper, contribution: hyper.scaled_noise_from_contributions(noise, contribution),
                             hyper_galaxies, contributions))
    return noise + sum(scaled_noises)


def evidence_from_reconstruction_terms(chi_squared_term, regularization_term,
                                       log_covariance_regularization_term,
                                       log_regularization_term,
                                       noise_term):
    return -0.5 * (chi_squared_term + regularization_term + log_covariance_regularization_term -
                   log_regularization_term + noise_term)


class FitterPositions:

    def __init__(self, positions, noise):

        self.positions = positions
        self.noise = noise


    @property
    def chi_squareds(self):
        return np.square(np.divide(self.maximum_separations, self.noise))

    @property
    def likelihood(self):
        return -0.5*sum(self.chi_squareds)

    def maximum_separation_within_threshold(self, threshold):
        if max(self.maximum_separations) > threshold:
            return False
        else:
            return True

    @property
    def maximum_separations(self):
        return list(map(lambda positions : self.max_separation_of_grid(positions), self.positions))

    def max_separation_of_grid(self, grid):
        rdist_max = np.zeros((grid.shape[0]))
        for i in range(grid.shape[0]):
            xdists = np.square(np.subtract(grid[i,0], grid[:,0]))
            ydists = np.square(np.subtract(grid[i,1], grid[:,1]))
            rdist_max[i] = np.max(np.add(xdists, ydists))
        return np.max(np.sqrt(rdist_max))