import numpy as np
from typing import Optional

import autoarray as aa


class Preloads(aa.Preloads):
    def __init__(
        self,
        blurred_image=None,
        sparse_grids_of_planes=None,
        mapper=None,
        blurred_mapping_matrix=None,
        curvature_matrix_sparse_preload=None,
        curvature_matrix_preload_counts=None,
        w_tilde=None,
        use_w_tilde=None,
    ):
        """
        Class which offers a concise API for settings up the preloads, which before a model-fit are set up via
        a comparison of two fits using two different models. If a quantity in these two fits is identical, it does
        not change thoughoutt he model-fit and can therefore be preloaded to avoid computation, speeding up
        the analysis.

        For example, the image-plane source-plane pixelization grid (which may be computationally expensive to compute
        via a KMeans algorithm) does not change for the majority of model-fits, because the assoicated model parameters
        are fixed. Preloading avoids reruning the KMeans algorithm for every model fitted, by preloading it in memory
        and using this preload in every fit.

        Parameters
        ----------

        Returns
        -------
        Preloads
            The preloads object used to skip certain calculations in the log likelihood function.

        """
        super().__init__(
            sparse_grids_of_planes=sparse_grids_of_planes,
            mapper=mapper,
            blurred_mapping_matrix=blurred_mapping_matrix,
            curvature_matrix_sparse_preload=curvature_matrix_sparse_preload,
            curvature_matrix_preload_counts=curvature_matrix_preload_counts,
            w_tilde=w_tilde,
            use_w_tilde=use_w_tilde,
        )

        self.blurred_image = blurred_image

    def set_sparse_grid_of_planes(self, fit_0, fit_1):
        """
        If a model contains a `Pixelization` that is an `instance` whose parameters are fixed and the `Result` contains
        the grid of this pixelization corresponding to these parameters, the grid can be preloaded to avoid repeating
        calculations which recompute the pixelization grid every iteration of the log likelihood function.

        This function inspects the `Result` and `Model` and returns a `Preloads` object with the appropriate pixelization
        grid for preloading. It raises an error if the `Model` is not suited to the preloading.

        Parameters
        ----------
        result
            The result containing the pixelization grid which is to be preloaded (corresponding to the maximum likelihood
            model of the model-fit).
        model
            The model, which is inspected to make sure the model-fit can have its pixelization preloaded.

        Returns
        -------
        Preloads
            The `Preloads` object containing the  (y,x) grid of coordinates representing the source plane pixelization
            centres.

        """

        sparse_image_plane_grid_of_planes_0 = fit_0.tracer.sparse_image_plane_grids_of_planes_from_grid(
            grid=fit_0.dataset.grid_inversion
        )

        sparse_image_plane_grid_of_planes_1 = fit_1.tracer.sparse_image_plane_grids_of_planes_from_grid(
            grid=fit_1.dataset.grid_inversion
        )

        if sparse_image_plane_grid_of_planes_0[-1] is not None:

            if np.allclose(
                sparse_image_plane_grid_of_planes_0[-1],
                sparse_image_plane_grid_of_planes_1[-1],
            ):

                self.sparse_grids_of_planes = sparse_image_plane_grid_of_planes_0
                return

        self.sparse_grids_of_planes = None

    def set_mapper(self, fit_0, fit_1):
        """
        If the `MassProfile`'s in a model are all fixed parameters, and the parameters of the source `Pixelization` are
        also fixed, the mapping of image-pixels to the source-pixels does not change for every likelihood evaluations.
        Matrices used by the linear algebra calculation in an `Inversion` can therefore be preloaded.

        This function inspects the `Result` and `Model` and returns a `Preload` object with the correct quantities for p
        reloading. It raises an error if the `Model` is not suited to the preloading.

        The preload is typically used when the lens light is being fitted, and a fixed mass model and source pixelization
        and regularization are being used. This occurs in the LIGHT PIPELINE of the SLaM pipelines.

        Parameters
        ----------
        result
            The result containing the linear algebra matrices which are to be preloaded (corresponding to the maximum
            likelihood model of the model-fit).
        model
            The model, which is inspected to make sure the model-fit can have its `Inversion` quantities preloaded.

        Returns
        -------
        Grid2D
            The `Preloads` object containing the `Inversion` linear algebra matrices.
        """

        if fit_0.inversion is None:

            self.mapper = None

            return

        mapper_0 = fit_0.inversion.mapper
        mapper_1 = fit_1.inversion.mapper

        if np.allclose(mapper_0.mapping_matrix, mapper_1.mapping_matrix):

            self.mapper = mapper_0

            return

        self.mapper = None

    def set_inversion(self, fit_0, fit_1):
        """
        If the `MassProfile`'s in a model are all fixed parameters, and the parameters of the source `Pixelization` are
        also fixed, the mapping of image-pixels to the source-pixels does not change for every likelihood evaluations.
        Matrices used by the linear algebra calculation in an `Inversion` can therefore be preloaded.

        This function inspects the `Result` and `Model` and returns a `Preload` object with the correct quantities for p
        reloading. It raises an error if the `Model` is not suited to the preloading.

        The preload is typically used when the lens light is being fitted, and a fixed mass model and source pixelization
        and regularization are being used. This occurs in the LIGHT PIPELINE of the SLaM pipelines.

        Parameters
        ----------
        result
            The result containing the linear algebra matrices which are to be preloaded (corresponding to the maximum
            likelihood model of the model-fit).
        model
            The model, which is inspected to make sure the model-fit can have its `Inversion` quantities preloaded.

        Returns
        -------
        Grid2D
            The `Preloads` object containing the `Inversion` linear algebra matrices.
        """
        inversion_0 = fit_0.inversion
        inversion_1 = fit_1.inversion

        return aa.Preloads(
            blurred_mapping_matrix=inversion.blurred_mapping_matrix,
            curvature_matrix_sparse_preload=inversion.curvature_matrix_sparse_preload,
            curvature_matrix_preload_counts=inversion.curvature_matrix_preload_counts,
            mapper=inversion.mapper,
            use_w_tilde=False,
        )

    def set_w_tilde(self, fit_0, fit_1):

        if fit_0.inversion is not None and np.allclose(
            fit_0.noise_map, fit_1.noise_map
        ):

            preload, indexes, lengths = aa.util.inversion.w_tilde_curvature_preload_imaging_from(
                noise_map_native=fit_0.noise_map.native,
                kernel_native=fit_0.dataset.psf.native,
                native_index_for_slim_index=fit_0.dataset.mask._native_index_for_slim_index,
            )

            w_tilde = aa.WTildeImaging(
                curvature_preload=preload,
                indexes=indexes.astype("int"),
                lengths=lengths.astype("int"),
                noise_map_value=fit_0.noise_map[0],
            )

            self.w_tilde = w_tilde
            self.use_w_tilde = True

            return

        self.w_tilde = None
        self.use_w_tilde = False