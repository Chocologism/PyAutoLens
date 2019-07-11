import numpy as np

import autofit as af
from autolens.model.galaxy import galaxy as g
from autolens.data.array.util import binning_util


class MockAnalysis(object):

    def __init__(self, number_galaxies, value):
        self.number_galaxies = number_galaxies
        self.value = value

    # noinspection PyUnusedLocal
    def galaxy_images_for_model(self, model):
        return self.number_galaxies * [np.array([self.value])]

    def fit(self, instance):
        return 1


class MockResults(object):

    def __init__(self, model_image=None, galaxy_images=(), constant=None, analysis=None,
                 optimizer=None):
        self.model_image = model_image
        self.unmasked_model_image = model_image
        self.galaxy_images = galaxy_images
        self.constant = constant or af.ModelInstance()
        self.variable = af.ModelMapper()
        self.analysis = analysis
        self.optimizer = optimizer

    @property
    def path_galaxy_tuples(self) -> [(str, g.Galaxy)]:
        """
        Tuples associating the names of galaxies with instances from the best fit
        """
        return [(('g0',), g.Galaxy(redshift=0.5)), (('g1',), g.Galaxy(redshift=1.0))]

    @property
    def path_galaxy_tuples_with_index(self) -> [(str, g.Galaxy)]:
        """
        Tuples associating the names of galaxies with instances from the best fit
        """
        return [(0, ('g0',), g.Galaxy(redshift=0.5)),
                (1, ('g1',), g.Galaxy(redshift=1.0))]

    @property
    def image_2d_dict(self) -> {str: g.Galaxy}:
        """
        A dictionary associating galaxy names with model images of those galaxies
        """
        return {galaxy_path: self.galaxy_images[i]
                for i, galaxy_path, galaxy
                in self.path_galaxy_tuples_with_index}

    def image_1d_dict_from_mask(self, mask) -> {str: g.Galaxy}:
        """
        A dictionary associating galaxy names with model images of those galaxies
        """

        image_1d_dict = {}

        for galaxy, galaxy_image_2d in self.image_2d_dict.items():

            image_1d_dict[galaxy] = mask.array_1d_from_array_2d(array_2d=galaxy_image_2d)

        return image_1d_dict

    def hyper_galaxy_image_1d_path_dict_from_mask(self, mask):
        """
        A dictionary associating 1D hyper galaxy images with their names.
        """

        hyper_minimum_percent = \
            af.conf.instance.general.get('hyper', 'hyper_minimum_percent', float)

        image_galaxy_1d_dict = self.image_1d_dict_from_mask(mask=mask)

        hyper_galaxy_image_1d_path_dict = {}

        for path, galaxy in self.path_galaxy_tuples:

            galaxy_image_1d = image_galaxy_1d_dict[path]

            minimum_galaxy_value = hyper_minimum_percent * max(galaxy_image_1d)
            galaxy_image_1d[galaxy_image_1d < minimum_galaxy_value] = minimum_galaxy_value
            hyper_galaxy_image_1d_path_dict[path] = galaxy_image_1d

        return hyper_galaxy_image_1d_path_dict

    def hyper_galaxy_image_2d_path_dict_from_mask(self, mask):
        """
        A dictionary associating 2D hyper galaxy images with their names.
        """

        hyper_galaxy_image_1d_path_dict = self.hyper_galaxy_image_1d_path_dict_from_mask(mask=mask)

        hyper_galaxy_image_2d_path_dict = {}

        for path, galaxy in self.path_galaxy_tuples:

            hyper_galaxy_image_2d_path_dict[path] = \
                mask.scaled_array_2d_from_array_1d(array_1d=hyper_galaxy_image_1d_path_dict[path])

        return hyper_galaxy_image_2d_path_dict

    def cluster_image_1d_dict_from_cluster(self, cluster) -> {str: g.Galaxy}:
        """
        A dictionary associating 1D cluster images with their names.
        """

        cluster_image_1d_dict = {}

        for galaxy, galaxy_image_2d in self.image_2d_dict.items():

            cluster_image_2d = binning_util.binned_up_array_2d_using_mean_from_array_2d_and_bin_up_factor(
                array_2d=galaxy_image_2d, bin_up_factor=cluster.bin_up_factor)

            cluster_image_1d_dict[galaxy] = \
                cluster.mask.array_1d_from_array_2d(array_2d=cluster_image_2d)

        return cluster_image_1d_dict

    def hyper_galaxy_cluster_image_1d_path_dict_from_cluster(self, cluster):
        """
        A dictionary associating 1D hyper galaxy cluster images with their names.
        """

        if cluster is not None:

            hyper_minimum_percent = \
                af.conf.instance.general.get('hyper', 'hyper_minimum_percent', float)

            cluster_image_1d_galaxy_dict = self.cluster_image_1d_dict_from_cluster(cluster=cluster)

            hyper_galaxy_cluster_image_path_dict = {}

            for path, galaxy in self.path_galaxy_tuples:
                galaxy_cluster_image_1d = cluster_image_1d_galaxy_dict[path]

                minimum_cluster_value = hyper_minimum_percent * max(galaxy_cluster_image_1d)
                galaxy_cluster_image_1d[galaxy_cluster_image_1d < minimum_cluster_value] = minimum_cluster_value

                hyper_galaxy_cluster_image_path_dict[path] = galaxy_cluster_image_1d

            return hyper_galaxy_cluster_image_path_dict

    def hyper_galaxy_cluster_image_2d_path_dict_from_cluster(
            self, cluster):
        """
        A dictionary associating "D hyper galaxy images cluster images with their names.
        """

        if cluster is not None:

            hyper_galaxy_cluster_image_1d_path_dict = \
                self.hyper_galaxy_cluster_image_1d_path_dict_from_cluster(cluster=cluster)

            hyper_galaxy_cluster_image_2d_path_dict = {}

            for path, galaxy in self.path_galaxy_tuples:
                hyper_galaxy_cluster_image_2d_path_dict[path] = \
                    cluster.mask.scaled_array_2d_from_array_1d(array_1d=hyper_galaxy_cluster_image_1d_path_dict[path])

            return hyper_galaxy_cluster_image_2d_path_dict

    def hyper_model_image_1d_from_mask(self, mask):

        hyper_galaxy_image_1d_path_dict = self.hyper_galaxy_image_1d_path_dict_from_mask(mask=mask)

        hyper_model_image_1d = np.zeros(mask.pixels_in_mask)

        for path, galaxy in self.path_galaxy_tuples:
            hyper_model_image_1d += hyper_galaxy_image_1d_path_dict[path]

        return hyper_model_image_1d


class MockResult:
    def __init__(self, constant, figure_of_merit, variable=None):
        self.constant = constant
        self.figure_of_merit = figure_of_merit
        self.variable = variable
        self.previous_variable = variable
        self.gaussian_tuples = None


class MockNLO(af.NonLinearOptimizer):

    def fit(self, analysis):
        class Fitness(object):

            def __init__(self, instance_from_physical_vector):
                self.result = None
                self.instance_from_physical_vector = instance_from_physical_vector

            def __call__(self, vector):
                instance = self.instance_from_physical_vector(vector)

                likelihood = analysis.fit(instance)
                self.result = MockResult(instance, likelihood)

                # Return Chi squared
                return -2 * likelihood

        fitness_function = Fitness(self.variable.instance_from_physical_vector)
        fitness_function(self.variable.prior_count * [0.8])

        return fitness_function.result