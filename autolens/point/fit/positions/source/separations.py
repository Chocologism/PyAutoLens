from typing import Optional

import autoarray as aa
import autogalaxy as ag

from autolens.lens.tracer import Tracer
from autolens.point.fit.positions.abstract import AbstractFitPositions
from autolens.point.solver import PointSolver

# from autoarray.fit import fit_util
from typing import Optional, Tuple
import numpy as np


class FitPositionsSource(AbstractFitPositions):
    def __init__(
        self,
        name: str,
        data: aa.Grid2DIrregular,
        noise_map: aa.ArrayIrregular,
        tracer: Tracer,
        solver: Optional[PointSolver],
        profile: Optional[ag.ps.Point] = None,
    ):
        """
        A lens position fitter, which takes a set of positions (e.g. from a plane in the tracer) and computes \
        their maximum separation, such that points which tracer closer to one another have a higher log_likelihood.

        Parameters
        ----------
        data : Grid2DIrregular
            The (y,x) arc-second coordinates of positions which the maximum distance and log_likelihood is computed using.
        noise_value
            The noise-value assumed when computing the log likelihood.
        """

        super().__init__(
            name=name,
            data=data,
            noise_map=noise_map,
            tracer=tracer,
            solver=solver,
            profile=profile,
        )

    @property
    def model_data(self) -> aa.Grid2DIrregular:
        """
        Returns the model positions, which are computed via the point solver.

        It if common for many more image-plane positions to be computed than actual positions in the dataset. In this
        case, each data point is paired with its closest model position.
        """
        if len(self.tracer.planes) <= 2:
            deflections = self.tracer.deflections_yx_2d_from(grid=self.data)
        else:
            deflections = self.tracer.deflections_between_planes_from(
                grid=self.data, plane_i=0, plane_j=self.source_plane_index
            )

        return self.data.grid_2d_via_deflection_grid_from(deflection_grid=deflections)

    @property
    def residual_map(self) -> aa.ArrayIrregular:
        return self.model_data.distances_to_coordinate_from(
            coordinate=self.source_plane_coordinate
        )

class FitPositionsSourceFast(AbstractFitPositions):
    def __init__(
        self,
        name: str,
        data: aa.Grid2DIrregular,
        noise_map: aa.ArrayIrregular,
        tracer: Tracer,
        solver: Optional[PointSolver],
        profile: Optional[ag.ps.Point] = None,
        scale : float = 0.04,
    ):
        """
        https://ui.adsabs.harvard.edu/abs/2020A%26A...636A..87C/abstract
        https://ui.adsabs.harvard.edu/link_gateway/2010PASJ...62.1017O/EPRINT_PDF
        Parameters
        ----------
        data : Grid2DIrregular
            The (y,x) arc-second coordinates of positions which the maximum distance and log_likelihood is computed using.
        noise_value
            The noise-value assumed when computing the log likelihood.
        scale
            buffer used in hessian magnification, correspondding to pixel scale of the dataset
        """
        super().__init__(
            name=name,
            data=data,
            noise_map=noise_map,
            tracer=tracer,
            solver=solver,
            profile=profile,
        )
        self.scale = scale
    @property
    def model_data(self) -> aa.Grid2DIrregular:
        """
        Returns the model positions, which are computed via the point solver.
        It if common for many more image-plane positions to be computed than actual positions in the dataset. In this
        case, each data point is paired with its closest model position.
        """
        if len(self.tracer.planes) <= 2:
            deflections = self.tracer.deflections_yx_2d_from(grid=self.data)
        else:
            deflections = self.tracer.deflections_between_planes_from(
                grid=self.data, plane_i=0, plane_j=self.source_plane_index
            )
        return self.data.grid_2d_via_deflection_grid_from(deflection_grid=deflections)
    
    @property
    def source_plane_coordinate(self) -> Tuple[float, float]:
        """
        https://ui.adsabs.harvard.edu/link_gateway/2010PASJ...62.1017O/EPRINT_PDF
        """
        model_data = self.model_data.array
        sqrt_mag_map = self.sqrt_magnification_map
        weight_sum = np.sum(sqrt_mag_map)
        centre_0 = np.sum(sqrt_mag_map * model_data[:,0]) / weight_sum
        centre_1 = np.sum(sqrt_mag_map * model_data[:,1]) / weight_sum
        return (centre_0,centre_1)
    
    @property
    def residual_map(self) -> aa.ArrayIrregular:
        return self.model_data.distances_to_coordinate_from(
            coordinate=self.source_plane_coordinate
        )
    
    @property
    def magnification_map(self) -> aa.Array2D:
        return self.tracer.magnification_2d_via_hessian_from(self.data, buffer=self.scale)
    @property
    def abs_magnification_map(self) -> np.ndarray:
        return np.abs(self.magnification_map.array)
    @property
    def sqrt_magnification_map(self) -> np.ndarray:
        return np.sqrt(self.abs_magnification_map)
    @property
    def noise_map(self):
        return self._noise_map / self.sqrt_magnification_map
    
class FitPositionsSourceFastWeighted(AbstractFitPositions):
    def __init__(
        self,
        name: str,
        data: aa.Grid2DIrregular,
        noise_map: aa.ArrayIrregular,
        tracer: Tracer,
        solver: Optional[PointSolver],
        profile: Optional[ag.ps.Point] = None,
        scale : float = 0.04,
        weight : float = 1.00,
    ):
        """
        https://ui.adsabs.harvard.edu/abs/2020A%26A...636A..87C/abstract
        https://ui.adsabs.harvard.edu/link_gateway/2010PASJ...62.1017O/EPRINT_PDF
        Parameters
        ----------
        data : Grid2DIrregular
            The (y,x) arc-second coordinates of positions which the maximum distance and log_likelihood is computed using.
        noise_value
            The noise-value assumed when computing the log likelihood.
        scale
            buffer used in hessian magnification, correspondding to pixel scale of the dataset
        """
        super().__init__(
            name=name,
            data=data,
            noise_map=noise_map,
            tracer=tracer,
            solver=solver,
            profile=profile,
        )
        self.scale = scale
        self.weight = weight
    @property
    def model_data(self) -> aa.Grid2DIrregular:
        """
        Returns the model positions, which are computed via the point solver.
        It if common for many more image-plane positions to be computed than actual positions in the dataset. In this
        case, each data point is paired with its closest model position.
        """
        if len(self.tracer.planes) <= 2:
            deflections = self.tracer.deflections_yx_2d_from(grid=self.data)
        else:
            deflections = self.tracer.deflections_between_planes_from(
                grid=self.data, plane_i=0, plane_j=self.source_plane_index
            )
        return self.data.grid_2d_via_deflection_grid_from(deflection_grid=deflections)
    
    @property
    def source_plane_coordinate(self) -> Tuple[float, float]:
        """
        https://ui.adsabs.harvard.edu/link_gateway/2010PASJ...62.1017O/EPRINT_PDF
        """
        model_data = self.model_data.array
        sqrt_mag_map = self.sqrt_magnification_map
        weight_sum = np.sum(sqrt_mag_map)
        centre_0 = np.sum(sqrt_mag_map * model_data[:,0]) / weight_sum
        centre_1 = np.sum(sqrt_mag_map * model_data[:,1]) / weight_sum
        return (centre_0,centre_1)
    
    @property
    def residual_map(self) -> aa.ArrayIrregular:
        return self.model_data.distances_to_coordinate_from(
            coordinate=self.source_plane_coordinate
        )
    
    @property
    def magnification_map(self) -> aa.Array2D:
        return self.tracer.magnification_2d_via_hessian_from(self.data, buffer=self.scale)
    @property
    def abs_magnification_map(self) -> np.ndarray:
        return np.abs(self.magnification_map.array)
    @property
    def sqrt_magnification_map(self) -> np.ndarray:
        return np.sqrt(self.abs_magnification_map)
    
    @property
    def sqrt_weight(self) -> float:
        return np.sqrt(self.weight)
    @property
    def noise_map(self):
        return self._noise_map / self.sqrt_magnification_map / self.sqrt_weight

class FitPositionsSourceFastReduced(AbstractFitPositions):
    def __init__(
        self,
        name: str,
        data: aa.Grid2DIrregular,
        noise_map: aa.ArrayIrregular,
        tracer: Tracer,
        solver: Optional[PointSolver],
        profile: Optional[ag.ps.Point] = None,
        scale : float = 0.04,
    ):
        """
        https://ui.adsabs.harvard.edu/abs/2020A%26A...636A..87C/abstract
        https://ui.adsabs.harvard.edu/link_gateway/2010PASJ...62.1017O/EPRINT_PDF

        Parameters
        ----------
        data : Grid2DIrregular
            The (y,x) arc-second coordinates of positions which the maximum distance and log_likelihood is computed using.
        noise_value
            The noise-value assumed when computing the log likelihood.
        scale
            buffer used in hessian magnification, correspondding to pixel scale of the dataset
        """

        super().__init__(
            name=name,
            data=data,
            noise_map=noise_map,
            tracer=tracer,
            solver=solver,
            profile=profile,
        )
        self.scale = scale
        
    @property
    def model_data(self) -> aa.Grid2DIrregular:
        """
        Returns the model positions, which are computed via the point solver.
        It if common for many more image-plane positions to be computed than actual positions in the dataset. In this
        case, each data point is paired with its closest model position.
        """
        if len(self.tracer.planes) <= 2:
            deflections = self.tracer.deflections_yx_2d_from(grid=self.data)
        else:
            deflections = self.tracer.deflections_between_planes_from(
                grid=self.data, plane_i=0, plane_j=self.source_plane_index
            )
        return self.data.grid_2d_via_deflection_grid_from(deflection_grid=deflections)
    
    @property
    def source_plane_coordinate(self) -> Tuple[float, float]:
        """
        https://ui.adsabs.harvard.edu/link_gateway/2010PASJ...62.1017O/EPRINT_PDF
        """
        model_data = self.model_data.array
        sqrt_mag_map = self.sqrt_magnification_map
        weight_sum = np.sum(sqrt_mag_map)
        centre_0 = np.sum(sqrt_mag_map * model_data[:,0]) / weight_sum
        centre_1 = np.sum(sqrt_mag_map * model_data[:,1]) / weight_sum
        return (centre_0,centre_1)
    
    @property
    def residual_map(self) -> aa.ArrayIrregular:
        return self.model_data.distances_to_coordinate_from(
            coordinate=self.source_plane_coordinate
        )
    
    @property
    def magnification_map(self) -> aa.Array2D:
        return self.tracer.magnification_2d_via_hessian_from(self.data, buffer=self.scale)
    @property
    def abs_magnification_map(self) -> np.ndarray:
        return np.abs(self.magnification_map.array)
    @property
    def sqrt_magnification_map(self) -> np.ndarray:
        return np.sqrt(self.abs_magnification_map)
    
    @property
    def noise_map(self):
        return self._noise_map / self.sqrt_magnification_map
    
    @property
    def reduced_chi_squared(self) -> float:
        """
        Returns the chi-squared terms of the model data's fit to an dataset, by summing the chi-squared-map.
        """
        return self.chi_squared/len(self.data.array)
    @property
    def log_likelihood(self) -> float:
        """
        Returns the log likelihood of each model data point's fit to the dataset, where:
        Log Likelihood = -0.5*[Chi_Squared_Term + Noise_Term] (see functions above for these definitions)
        """
        # return fit_util.log_likelihood_from(
        #     chi_squared=self.reduced_chi_squared, noise_normalization=0.0
        # )
        return -0.5 * self.reduced_chi_squared