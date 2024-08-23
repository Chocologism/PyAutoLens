import pytest

import autolens as al
import autogalaxy as ag
from autolens.point.solver.circle_solver import CircleSolver
from autolens.point.visualise import visualise


@pytest.fixture
def solver(grid):
    return CircleSolver.for_grid(
        grid=grid,
        pixel_scale_precision=0.001,
    )


def test_solver_basic(solver):
    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                mass=ag.mp.Isothermal(
                    centre=(0.0, 0.0),
                    einstein_radius=1.0,
                ),
            ),
            al.Galaxy(
                redshift=1.0,
            ),
        ]
    )

    for step in solver.steps(
        tracer=tracer,
        source_plane_coordinate=(0.0, 0.0),
        radius=0.01,
    ):
        visualise(step)
