import pytest

import autolens as al


@pytest.fixture(name="tracer")
def make_tracer():
    mass = al.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.1, 0.05), einstein_radius=1.6
    )

    lens_galaxy = al.Galaxy(redshift=0.5, mass=mass)

    disk = al.lp.Exponential(
        centre=(0.3, 0.2),
        ell_comps=(0.05, 0.25),
        intensity=0.05,
        effective_radius=0.5,
    )

    source_galaxy = al.Galaxy(redshift=1.0, disk=disk)

    return al.Tracer.from_galaxies(
        galaxies=[lens_galaxy, source_galaxy], cosmology=al.cosmo.wrap.Planck15
    )


@pytest.fixture(name="tracer_dict")
def make_tracer_dict():
    return {
        "arguments": {
            "cosmology": "Planck15",
            "planes": [
                {
                    "arguments": {
                        "galaxies": [
                            {
                                "arguments": {
                                    "mass": {
                                        "arguments": {
                                            "centre": (0.0, 0.0),
                                            "einstein_radius": 1.6,
                                            "ell_comps": (0.1, 0.05),
                                        },
                                        "type": "instance",
                                        "class_path": "autogalaxy.profiles.mass.total.Isothermal",
                                    },
                                    "pixelization": None,
                                    "redshift": 0.5,
                                    "regularization": None,
                                },
                                "type": "instance",
                                "class_path": "autogalaxy.galaxy.galaxy.Galaxy",
                            }
                        ],
                        "run_time_dict": None,
                        "redshift": 0.5,
                    },
                    "type": "instance",
                    "class_path": "autogalaxy.plane.plane.Plane",
                },
                {
                    "arguments": {
                        "galaxies": [
                            {
                                "arguments": {
                                    "disk": {
                                        "arguments": {
                                            "centre": (0.3, 0.2),
                                            "effective_radius": 0.5,
                                            "ell_comps": (0.05, 0.25),
                                            "intensity": 0.05,
                                        },
                                        "type": "instance",
                                        "class_path": "autogalaxy.profiles.light.standard.Exponential",
                                    },
                                    "pixelization": None,
                                    "redshift": 1.0,
                                    "regularization": None,
                                },
                                "type": "instance",
                                "class_path": "autogalaxy.galaxy.galaxy.Galaxy",
                            }
                        ],
                        "run_time_dict": None,
                        "redshift": 1.0,
                    },
                    "type": "instance",
                    "class_path": "autogalaxy.plane.plane.Plane",
                },
            ],
            "run_time_dict": None,
        },
        "type": "instance",
        "class_path": "autolens.lens.ray_tracing.Tracer",
    }


# def test_to_dict(tracer, tracer_dict):
#     assert tracer.dict() == tracer_dict


def test_from_dict(tracer, tracer_dict):
    assert tracer.from_dict(tracer_dict) == tracer
