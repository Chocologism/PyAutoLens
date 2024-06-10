import math
from typing import Tuple

from autoarray import Grid2D, Grid2DIrregular
from autoarray.structures.triangles.subsample_triangles import SubsampleTriangles
from autoarray.structures.triangles.triangles import Triangles
from autoarray.type import Grid2DLike
from autogalaxy import OperateDeflections


class TriangleSolver:
    def __init__(
        self,
        lensing_obj: OperateDeflections,
        grid: Grid2D,
        pixel_scale_precision: float,
    ):
        """
        Determine the image plane coordinates that are traced to be a source plane coordinate.

        This is performed efficiently by iteratively subdividing the image plane into triangles and checking if the
        source plane coordinate is contained within the triangle. The triangles are subsampled to increase the
        resolution

        Parameters
        ----------
        lensing_obj
            A tracer describing the lensing system.
        grid
            The grid of image plane coordinates.
        pixel_scale_precision
            The target pixel scale of the image grid.
        """
        self.lensing_obj = lensing_obj
        self.grid = grid
        self.pixel_scale_precision = pixel_scale_precision

    @property
    def n_steps(self) -> int:
        """
        How many times should triangles be subdivided?
        """
        return math.ceil(math.log2(self.grid.pixel_scale / self.pixel_scale_precision))

    def _source_plane_grid(self, grid: Grid2DLike) -> Grid2DLike:
        """
        Calculate the source plane grid from the image plane grid.

        Parameters
        ----------
        grid
            The image plane grid.

        Returns
        -------
        The source plane grid computed by applying the deflections to the image plane grid.
        """
        deflections = self.lensing_obj.deflections_yx_2d_from(grid=grid)
        # noinspection PyTypeChecker
        return grid.grid_2d_via_deflection_grid_from(deflection_grid=deflections)

    def solve(self, source_plane_coordinate: Tuple[float, float]) -> Grid2DIrregular:
        """
        Solve for the image plane coordinates that are traced to the source plane coordinate.

        Parameters
        ----------
        source_plane_coordinate
            The source plane coordinate to trace to the image plane.

        Returns
        -------
        A list of image plane coordinates that are traced to the source plane coordinate.
        """
        triangles = Triangles.for_grid(grid=self.grid)

        if self.n_steps == 0:
            raise ValueError(
                "The target pixel scale is too large to subdivide the triangles."
            )

        kept_triangles = []

        for _ in range(self.n_steps):
            kept_triangles = self._filter_triangles(
                triangles=triangles,
                source_plane_coordinate=source_plane_coordinate,
            )
            with_neighbourhood = {
                triangle
                for kept_triangle in kept_triangles
                for triangle in kept_triangle.neighbourhood
            }
            triangles = SubsampleTriangles(parent_triangles=list(with_neighbourhood))

        return Grid2DIrregular([triangle.mean for triangle in kept_triangles])

    def _filter_triangles(
        self,
        triangles: Triangles,
        source_plane_coordinate: Tuple[float, float],
    ):
        """
        Filter the triangles to keep only those that contain the source plane coordinate.

        Parameters
        ----------
        triangles
            A set of triangles that may contain the source plane coordinate.
        source_plane_coordinate
            The source plane coordinate to check if it is contained within the triangles.

        Returns
        -------
        The triangles that contain the source plane coordinate.
        """
        source_plane_grid = self._source_plane_grid(grid=triangles.grid_2d)

        kept_triangles = []
        for image_triangle, source_triangle in zip(
            triangles.triangles,
            triangles.with_updated_grid(source_plane_grid),
        ):
            if source_triangle.contains(
                point=source_plane_coordinate,
            ):
                kept_triangles.append(image_triangle)

        return kept_triangles
