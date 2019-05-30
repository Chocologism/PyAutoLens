import numpy as np
import pytest
from astropy import cosmology as cosmo

from autolens.data.array import grids
from autolens.data.array import mask as msk
from autolens.data import ccd
from autolens.data import convolution
from autolens.lens import plane as pl
from autolens.lens import ray_tracing
from autolens.lens.util import lens_util
from autolens.model import cosmology_util
from autolens.model.galaxy import galaxy as g
from autolens.model.galaxy.util import galaxy_util
from autolens.model.inversion import pixelizations, regularization
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from test.unit.mock.mock_grids import MockBorders
from test.unit.mock.mock_inversion import MockRegularization, MockPixelization


@pytest.fixture(name="grid_stack")
def make_grid_stack():
    mask = msk.Mask(np.array([[True, True, True, True],
                              [True, False, False, True],
                              [True, True, True, True]]), pixel_scale=6.0)

    grid_stack = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask, sub_grid_size=2,
                                                                                  psf_shape=(3, 3))

    # Manually overwrite a set of coordinates to make tests of grid_stacks and deflections straightforward

    grid_stack.regular[0] = np.array([1.0, 1.0])
    grid_stack.regular[1] = np.array([1.0, 0.0])
    grid_stack.sub[0] = np.array([1.0, 1.0])
    grid_stack.sub[1] = np.array([1.0, 0.0])
    grid_stack.sub[2] = np.array([1.0, 1.0])
    grid_stack.sub[3] = np.array([1.0, 0.0])
    grid_stack.blurring[0] = np.array([1.0, 0.0])

    return grid_stack


@pytest.fixture(name="padded_grid_stack")
def make_padded_grid_stack():
    mask = msk.Mask(np.array([[True, False]]), pixel_scale=3.0)
    return grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(mask, 2, (3, 3))


@pytest.fixture(name='galaxy_non', scope='function')
def make_galaxy_non():
    return g.Galaxy(redshift=0.5)


@pytest.fixture(name="galaxy_light")
def make_galaxy_light():
    return g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(centre=(0.1, 0.1), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                      effective_radius=0.6, sersic_index=4.0))


@pytest.fixture(name="galaxy_mass")
def make_galaxy_mass():
    return g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))


@pytest.fixture(name='galaxy_mass_x2')
def make_galaxy_mass_x2():
    return g.Galaxy(sis_0=mp.SphericalIsothermal(einstein_radius=1.0),
                    sis_1=mp.SphericalIsothermal(einstein_radius=1.0))


class TestAbstractTracer(object):

    class TestProperties:

        def test__total_planes(self, grid_stack):

            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g.Galaxy(redshift=0.5)], image_plane_grid_stack=grid_stack)

            assert tracer.total_planes == 1

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g.Galaxy(redshift=0.5)], source_galaxies=[g.Galaxy(redshift=0.5)],
                                                         image_plane_grid_stack=grid_stack)

            assert tracer.total_planes == 2

            tracer = ray_tracing.TracerMultiPlanes(galaxies=[g.Galaxy(redshift=1.0), g.Galaxy(redshift=2.0),
                                                             g.Galaxy(redshift=3.0)],
                                                   image_plane_grid_stack=grid_stack)

            assert tracer.total_planes == 3

            tracer = ray_tracing.TracerMultiPlanes(galaxies=[g.Galaxy(redshift=1.0), g.Galaxy(redshift=2.0),
                                                             g.Galaxy(redshift=1.0)],
                                                   image_plane_grid_stack=grid_stack)

            assert tracer.total_planes == 2

        def test__has_galaxy_with_light_profile(self, grid_stack):

            gal = g.Galaxy(redshift=0.5)
            gal_lp = g.Galaxy(redshift=0.5, light_profile=lp.LightProfile())
            gal_mp = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal())

            assert ray_tracing.TracerImageSourcePlanes([gal], [gal],
                                                       image_plane_grid_stack=grid_stack).has_light_profile is False
            assert ray_tracing.TracerImageSourcePlanes([gal_mp], [gal_mp],
                                                       image_plane_grid_stack=grid_stack).has_light_profile is False
            assert ray_tracing.TracerImageSourcePlanes([gal_lp], [gal_lp],
                                                       image_plane_grid_stack=grid_stack).has_light_profile is True
            assert ray_tracing.TracerImageSourcePlanes([gal_lp], [gal],
                                                       image_plane_grid_stack=grid_stack).has_light_profile is True
            assert ray_tracing.TracerImageSourcePlanes([gal_lp], [gal_mp],
                                                       image_plane_grid_stack=grid_stack).has_light_profile is True

        def test_plane_with_galaxy(self, grid_stack):
            g1 = g.Galaxy(redshift=1)
            g2 = g.Galaxy(redshift=2)

            tracer = ray_tracing.TracerImageSourcePlanes([g1], [g2], grid_stack)

            assert tracer.plane_with_galaxy(g1).galaxies == [g1]
            assert tracer.plane_with_galaxy(g2).galaxies == [g2]

        def test__has_galaxy_with_mass_profile(self, grid_stack):
            gal = g.Galaxy(redshift=0.5)
            gal_lp = g.Galaxy(redshift=0.5, light_profile=lp.LightProfile())
            gal_mp = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal())

            assert ray_tracing.TracerImageSourcePlanes([gal], [gal],
                                                       image_plane_grid_stack=grid_stack).has_mass_profile is False
            assert ray_tracing.TracerImageSourcePlanes([gal_mp], [gal_mp],
                                                       image_plane_grid_stack=grid_stack).has_mass_profile is True
            assert ray_tracing.TracerImageSourcePlanes([gal_lp], [gal_lp],
                                                       image_plane_grid_stack=grid_stack).has_mass_profile is False
            assert ray_tracing.TracerImageSourcePlanes([gal_lp], [gal],
                                                       image_plane_grid_stack=grid_stack).has_mass_profile is False
            assert ray_tracing.TracerImageSourcePlanes([gal_lp], [gal_mp],
                                                       image_plane_grid_stack=grid_stack).has_mass_profile is True

        def test__has_galaxy_with_pixelization(self, grid_stack):
            gal = g.Galaxy(redshift=0.5)
            gal_lp = g.Galaxy(redshift=0.5, light_profile=lp.LightProfile())
            gal_pix = g.Galaxy(redshift=0.5, pixelization=pixelizations.Pixelization(), regularization=regularization.Constant())

            assert ray_tracing.TracerImageSourcePlanes([gal], [gal],
                                                       image_plane_grid_stack=grid_stack).has_pixelization is False
            assert ray_tracing.TracerImageSourcePlanes([gal_lp], [gal_lp],
                                                       image_plane_grid_stack=grid_stack).has_pixelization is False
            assert ray_tracing.TracerImageSourcePlanes([gal_pix], [gal_pix],
                                                       image_plane_grid_stack=grid_stack).has_pixelization is True
            assert ray_tracing.TracerImageSourcePlanes([gal_pix], [gal],
                                                       image_plane_grid_stack=grid_stack).has_pixelization is True
            assert ray_tracing.TracerImageSourcePlanes([gal_pix], [gal_lp],
                                                       image_plane_grid_stack=grid_stack).has_pixelization is True

        def test__has_galaxy_with_regularization(self, grid_stack):
            gal = g.Galaxy(redshift=0.5)
            gal_lp = g.Galaxy(redshift=0.5, light_profile=lp.LightProfile())
            gal_reg = g.Galaxy(redshift=0.5, pixelization=pixelizations.Pixelization(), regularization=regularization.Constant())

            assert ray_tracing.TracerImageSourcePlanes([gal], [gal],
                                                       image_plane_grid_stack=grid_stack).has_regularization is False
            assert ray_tracing.TracerImageSourcePlanes([gal_lp], [gal_lp],
                                                       image_plane_grid_stack=grid_stack).has_regularization is False
            assert ray_tracing.TracerImageSourcePlanes([gal_reg], [gal_reg],
                                                       image_plane_grid_stack=grid_stack).has_regularization is True
            assert ray_tracing.TracerImageSourcePlanes([gal_reg], [gal],
                                                       image_plane_grid_stack=grid_stack).has_regularization is True
            assert ray_tracing.TracerImageSourcePlanes([gal_reg], [gal_lp],
                                                       image_plane_grid_stack=grid_stack).has_regularization is True

        def test__has_galaxy_with_hyper_galaxy(self, grid_stack):

            gal = g.Galaxy(redshift=0.5)
            gal_lp = g.Galaxy(redshift=0.5, light_profile=lp.LightProfile())
            gal_hyper = g.Galaxy(redshift=0.5, hyper_galaxy=g.HyperGalaxy())

            assert ray_tracing.TracerImageSourcePlanes([gal], [gal],
                                                       image_plane_grid_stack=grid_stack).has_hyper_galaxy is False
            assert ray_tracing.TracerImageSourcePlanes([gal_lp], [gal_lp],
                                                       image_plane_grid_stack=grid_stack).has_hyper_galaxy is False
            assert ray_tracing.TracerImageSourcePlanes([gal_hyper], [gal_hyper],
                                                       image_plane_grid_stack=grid_stack).has_hyper_galaxy is True
            assert ray_tracing.TracerImageSourcePlanes([gal_hyper], [gal],
                                                       image_plane_grid_stack=grid_stack).has_hyper_galaxy is True
            assert ray_tracing.TracerImageSourcePlanes([gal_hyper], [gal_lp],
                                                       image_plane_grid_stack=grid_stack).has_hyper_galaxy is True

    class TestImages:

        def test__no_galaxy_has_light_profile__image_plane_is_returned_as_none(self, grid_stack):
            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g.Galaxy(redshift=0.5)], image_plane_grid_stack=grid_stack)

            assert tracer.profile_image_plane_image_2d is None
            assert tracer.profile_image_plane_image_2d_for_simulation is None
            assert tracer.profile_image_plane_image_1d is None
            assert tracer.profile_image_plane_blurring_image_1d is None

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g.Galaxy(redshift=0.5)], source_galaxies=[g.Galaxy(redshift=0.5)],
                                                         image_plane_grid_stack=grid_stack)

            assert tracer.profile_image_plane_image_2d is None
            assert tracer.profile_image_plane_image_2d_for_simulation is None
            assert tracer.profile_image_plane_image_1d is None
            assert tracer.profile_image_plane_blurring_image_1d is None

            tracer = ray_tracing.TracerMultiPlanes(galaxies=[g.Galaxy(redshift=0.1), g.Galaxy(redshift=0.2)],
                                                   image_plane_grid_stack=grid_stack)

            assert tracer.profile_image_plane_image_2d is None
            assert tracer.profile_image_plane_image_2d_for_simulation is None
            assert tracer.profile_image_plane_image_1d is None
            assert tracer.profile_image_plane_blurring_image_1d is None

    class TestConvergence:

        def test__galaxy_mass_sis__no_source_plane_convergence(self, grid_stack):
            g0 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(redshift=0.5)

            image_plane = pl.Plane(galaxies=[g0], grid_stack=grid_stack, compute_deflections=True)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                         image_plane_grid_stack=grid_stack)

            assert image_plane.convergence.shape == (3, 4)
            assert (image_plane.convergence == tracer.convergence).all()

        def test__galaxy_entered_3_times__both_planes__different_convergence_for_each(self, grid_stack):
            g0 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(einstein_radius=2.0))
            g2 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(einstein_radius=3.0))

            g0_convergence = galaxy_util.convergence_of_galaxies_from_grid(grid_stack.sub.unlensed_sub_grid,
                                                                           galaxies=[g0])
            g1_convergence = galaxy_util.convergence_of_galaxies_from_grid(grid_stack.sub.unlensed_sub_grid,
                                                                           galaxies=[g1])
            g2_convergence = galaxy_util.convergence_of_galaxies_from_grid(grid_stack.sub.unlensed_sub_grid,
                                                                           galaxies=[g2])

            g0_convergence = grid_stack.regular.scaled_array_2d_from_array_1d(g0_convergence)
            g1_convergence = grid_stack.regular.scaled_array_2d_from_array_1d(g1_convergence)
            g2_convergence = grid_stack.regular.scaled_array_2d_from_array_1d(g2_convergence)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1], source_galaxies=[g2],
                                                         image_plane_grid_stack=grid_stack)

            assert tracer.image_plane.convergence == pytest.approx(g0_convergence + g1_convergence, 1.0e-4)
            assert (tracer.source_plane.convergence == g2_convergence).all()
            assert tracer.convergence == pytest.approx(g0_convergence + g1_convergence +
                                                       g2_convergence, 1.0e-4)

        def test__padded_2d_convergence_from_plane__mapped_correctly(self, padded_grid_stack, galaxy_mass):
            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_mass], source_galaxies=[g.Galaxy(redshift=0.5)],
                                                         image_plane_grid_stack=padded_grid_stack)

            assert tracer.image_plane.convergence.shape == (1, 2)
            assert tracer.source_plane.convergence.shape == (1, 2)
            assert (tracer.image_plane.convergence == tracer.convergence).all()

        def test__no_galaxy_has_mass_profile__convergence_returned_as_none(self, grid_stack):
            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g.Galaxy(redshift=0.5)], image_plane_grid_stack=grid_stack)

            assert tracer.convergence is None

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g.Galaxy(redshift=0.5)], source_galaxies=[g.Galaxy(redshift=0.5)],
                                                         image_plane_grid_stack=grid_stack)

            assert tracer.convergence is None

            tracer = ray_tracing.TracerMultiPlanes(galaxies=[g.Galaxy(redshift=0.1), g.Galaxy(redshift=0.2)],
                                                   image_plane_grid_stack=grid_stack)

            assert tracer.convergence is None

    class TestPotential:

        def test__galaxy_mass_sis__source_plane_no_mass_potential_is_ignored(self, grid_stack):
            g0 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(redshift=0.5)

            image_plane = pl.Plane(galaxies=[g0], grid_stack=grid_stack, compute_deflections=True)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                         image_plane_grid_stack=grid_stack)

            assert tracer.potential.shape == (3, 4)
            assert (image_plane.potential == tracer.potential).all()

        def test__galaxy_entered_3_times__different_potential_for_each(self, grid_stack):
            g0 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(einstein_radius=2.0))
            g2 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(einstein_radius=3.0))

            g0_potential = galaxy_util.potential_of_galaxies_from_grid(grid_stack.sub.unlensed_sub_grid, galaxies=[g0])
            g1_potential = galaxy_util.potential_of_galaxies_from_grid(grid_stack.sub.unlensed_sub_grid, galaxies=[g1])
            g2_potential = galaxy_util.potential_of_galaxies_from_grid(grid_stack.sub.unlensed_sub_grid, galaxies=[g2])

            g0_potential = grid_stack.regular.scaled_array_2d_from_array_1d(g0_potential)
            g1_potential = grid_stack.regular.scaled_array_2d_from_array_1d(g1_potential)
            g2_potential = grid_stack.regular.scaled_array_2d_from_array_1d(g2_potential)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1], source_galaxies=[g2],
                                                         image_plane_grid_stack=grid_stack)

            assert tracer.image_plane.potential == pytest.approx(g0_potential + g1_potential, 1.0e-4)
            assert (tracer.source_plane.potential == g2_potential).all()
            assert tracer.potential == pytest.approx(g0_potential + g1_potential + g2_potential, 1.0e-4)

        def test__padded_2d_potential_from_plane__mapped_correctly(self, padded_grid_stack, galaxy_mass):
            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_mass], source_galaxies=[g.Galaxy(redshift=0.5)],
                                                         image_plane_grid_stack=padded_grid_stack)

            assert tracer.image_plane.potential.shape == (1, 2)
            assert tracer.source_plane.potential.shape == (1, 2)
            assert (tracer.image_plane.potential == tracer.potential).all()

        def test__no_galaxy_has_mass_profile__potential_returned_as_none(self, grid_stack):
            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g.Galaxy(redshift=0.5)], image_plane_grid_stack=grid_stack)

            assert tracer.potential is None

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g.Galaxy(redshift=0.5)], source_galaxies=[g.Galaxy(redshift=0.5)],
                                                         image_plane_grid_stack=grid_stack)

            assert tracer.potential is None

            tracer = ray_tracing.TracerMultiPlanes(galaxies=[g.Galaxy(redshift=0.1), g.Galaxy(redshift=0.2)],
                                                   image_plane_grid_stack=grid_stack)

            assert tracer.potential is None

    class TestDeflections:

        def test__galaxy_mass_sis__source_plane_no_mass__deflections_is_ignored(self, grid_stack):
            g0 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(redshift=0.5)

            image_plane = pl.Plane(galaxies=[g0], grid_stack=grid_stack, compute_deflections=True)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                         image_plane_grid_stack=grid_stack)

            assert tracer.deflections_y.shape == (3, 4)
            assert (image_plane.deflections_y == tracer.deflections_y).all()
            assert tracer.deflections_x.shape == (3, 4)
            assert (image_plane.deflections_x == tracer.deflections_x).all()

        def test__galaxy_entered_3_times__different_deflections_for_each(self, grid_stack):
            g0 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(einstein_radius=2.0))
            g2 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(einstein_radius=3.0))

            g0_deflections = galaxy_util.deflections_of_galaxies_from_grid(grid=grid_stack.sub.unlensed_sub_grid,
                                                                           galaxies=[g0])
            g1_deflections = galaxy_util.deflections_of_galaxies_from_grid(grid=grid_stack.sub.unlensed_sub_grid,
                                                                           galaxies=[g1])
            g2_deflections = galaxy_util.deflections_of_galaxies_from_grid(grid=grid_stack.sub.unlensed_sub_grid,
                                                                           galaxies=[g2])

            g0_deflections_y = grid_stack.regular.scaled_array_2d_from_array_1d(g0_deflections[:, 0])
            g1_deflections_y = grid_stack.regular.scaled_array_2d_from_array_1d(g1_deflections[:, 0])
            g2_deflections_y = grid_stack.regular.scaled_array_2d_from_array_1d(g2_deflections[:, 0])

            g0_deflections_x = grid_stack.regular.scaled_array_2d_from_array_1d(g0_deflections[:, 1])
            g1_deflections_x = grid_stack.regular.scaled_array_2d_from_array_1d(g1_deflections[:, 1])
            g2_deflections_x = grid_stack.regular.scaled_array_2d_from_array_1d(g2_deflections[:, 1])

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1], source_galaxies=[g2],
                                                         image_plane_grid_stack=grid_stack)

            assert tracer.image_plane.deflections_y == pytest.approx(g0_deflections_y + g1_deflections_y, 1.0e-4)
            assert (tracer.source_plane.deflections_y == g2_deflections_y).all()
            assert tracer.deflections_y == pytest.approx(g0_deflections_y + g1_deflections_y + g2_deflections_y, 1.0e-4)

            assert tracer.image_plane.deflections_x == pytest.approx(g0_deflections_x + g1_deflections_x, 1.0e-4)
            assert (tracer.source_plane.deflections_x == g2_deflections_x).all()
            assert tracer.deflections_x == pytest.approx(g0_deflections_x + g1_deflections_x + g2_deflections_x, 1.0e-4)

        def test__padded_2d_deflections_from_plane__mapped_correctly(self, padded_grid_stack, galaxy_mass):
            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_mass], source_galaxies=[g.Galaxy(redshift=0.5)],
                                                         image_plane_grid_stack=padded_grid_stack)

            assert tracer.image_plane.deflections_y.shape == (1, 2)
            assert tracer.source_plane.deflections_y.shape == (1, 2)
            assert (tracer.image_plane.deflections_y == tracer.deflections_y).all()

            assert tracer.image_plane.deflections_x.shape == (1, 2)
            assert tracer.source_plane.deflections_x.shape == (1, 2)
            assert (tracer.image_plane.deflections_x == tracer.deflections_x).all()

        def test__no_galaxy_has_mass_profile__deflections_returned_as_none(self, grid_stack):
            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g.Galaxy(redshift=0.5)], image_plane_grid_stack=grid_stack)

            assert tracer.deflections_y is None
            assert tracer.deflections_x is None

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g.Galaxy(redshift=0.5)], source_galaxies=[g.Galaxy(redshift=0.5)],
                                                         image_plane_grid_stack=grid_stack)

            assert tracer.deflections_y is None
            assert tracer.deflections_x is None

            tracer = ray_tracing.TracerMultiPlanes(galaxies=[g.Galaxy(redshift=0.1), g.Galaxy(redshift=0.2)],
                                                   image_plane_grid_stack=grid_stack)

            assert tracer.deflections_y is None
            assert tracer.deflections_x is None

    class TestMappers:

        def test__no_galaxy_has_pixelization__returns_empty_list(self, grid_stack):
            galaxy_no_pix = g.Galaxy(redshift=0.5)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_no_pix], source_galaxies=[galaxy_no_pix],
                                                         image_plane_grid_stack=grid_stack, border=[MockBorders()])

            assert tracer.mappers_of_planes == []

        def test__source_galaxy_has_pixelization__returns_mapper(self, grid_stack):
            galaxy_pix = g.Galaxy(redshift=0.5, pixelization=MockPixelization(value=1), regularization=MockRegularization(value=0))
            galaxy_no_pix = g.Galaxy(redshift=0.5)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_no_pix], source_galaxies=[galaxy_pix],
                                                         image_plane_grid_stack=grid_stack, border=[MockBorders()])

            assert tracer.mappers_of_planes[0] == 1

        def test__both_galaxies_have_pixelization__returns_both_mappers(self, grid_stack):
            galaxy_pix_0 = g.Galaxy(redshift=0.5, pixelization=MockPixelization(value=1), regularization=MockRegularization(value=3))
            galaxy_pix_1 = g.Galaxy(redshift=0.5, pixelization=MockPixelization(value=2), regularization=MockRegularization(value=4))

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_pix_0], source_galaxies=[galaxy_pix_1],
                                                         image_plane_grid_stack=grid_stack, border=[MockBorders()])

            assert tracer.mappers_of_planes[0] == 1
            assert tracer.mappers_of_planes[1] == 2

    class TestRegularizations:

        def test__no_galaxy_has_regularization__returns_empty_list(self, grid_stack):
            galaxy_no_reg = g.Galaxy(redshift=0.5)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_no_reg], source_galaxies=[galaxy_no_reg],
                                                         image_plane_grid_stack=grid_stack, border=MockBorders())

            assert tracer.regularizations_of_planes == []

        def test__source_galaxy_has_regularization__returns_regularizations(self, grid_stack):
            galaxy_reg = g.Galaxy(redshift=0.5, pixelization=MockPixelization(value=1), regularization=MockRegularization(value=0))
            galaxy_no_reg = g.Galaxy(redshift=0.5)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_no_reg], source_galaxies=[galaxy_reg],
                                                         image_plane_grid_stack=grid_stack, border=MockBorders())

            assert tracer.regularizations_of_planes[0].value == 0

        def test__both_galaxies_have_regularization__returns_both_regularizations(self, grid_stack):
            galaxy_reg_0 = g.Galaxy(redshift=0.5, pixelization=MockPixelization(value=1), regularization=MockRegularization(value=3))
            galaxy_reg_1 = g.Galaxy(redshift=0.5, pixelization=MockPixelization(value=2), regularization=MockRegularization(value=4))

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_reg_0], source_galaxies=[galaxy_reg_1],
                                                         image_plane_grid_stack=grid_stack, border=MockBorders())

            assert tracer.regularizations_of_planes[0].value == 3
            assert tracer.regularizations_of_planes[1].value == 4

    class TestCosmology:

        def test__2_planes__z01_and_z1(self, grid_stack):

            g0 = g.Galaxy(redshift=0.1)
            g1 = g.Galaxy(redshift=1.0)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                         image_plane_grid_stack=grid_stack,
                                                         cosmology=cosmo.Planck15)

            assert tracer.image_plane.arcsec_per_kpc == pytest.approx(0.525060, 1e-5)
            assert tracer.image_plane.kpc_per_arcsec == pytest.approx(1.904544, 1e-5)
            assert tracer.image_plane.angular_diameter_distance_to_earth_in_units(unit_length='kpc') == \
                   pytest.approx(392840, 1e-5)

            assert tracer.source_plane.arcsec_per_kpc == pytest.approx(0.1214785, 1e-5)
            assert tracer.source_plane.kpc_per_arcsec == pytest.approx(8.231907, 1e-5)
            assert tracer.source_plane.angular_diameter_distance_to_earth_in_units(unit_length='kpc') == \
                   pytest.approx(1697952, 1e-5)

            assert tracer.angular_diameter_distance_from_image_to_source_plane_in_units(unit_length='kpc') == \
                   pytest.approx(1481890.4, 1e-5)

            assert tracer.critical_surface_density_between_planes_in_units(i=0, j=1, unit_length='kpc',
                                                                                unit_mass='solMass') == \
                   pytest.approx(4.85e9, 1e-2)


            assert tracer.critical_surface_density_between_image_and_source_plane_in_units(unit_length='kpc',
                                                                                unit_mass='solMass') == \
                   pytest.approx(4.85e9, 1e-2)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                         image_plane_grid_stack=grid_stack,
                                                         cosmology=cosmo.Planck15)

            assert tracer.critical_surface_density_between_planes_in_units(i=0, j=1, unit_length='arcsec',
                                                                                unit_mass='solMass') == \
                   pytest.approx(17593241668, 1e-2)

    class TestGalaxyLists:

        def test__galaxy_list__comes_in_plane_redshift_order(self, grid_stack):
            g0 = g.Galaxy(redshift=0.5)
            g1 = g.Galaxy(redshift=0.5)

            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g0, g1], image_plane_grid_stack=grid_stack)

            assert tracer.galaxies == [g0, g1]

            g2 = g.Galaxy(redshift=1.0)
            g3 = g.Galaxy(redshift=1.0)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1], source_galaxies=[g2, g3],
                                                         image_plane_grid_stack=grid_stack)

            assert tracer.galaxies == [g0, g1, g2, g3]

            g4 = g.Galaxy(redshift=0.75)
            g5 = g.Galaxy(redshift=1.5)

            tracer = ray_tracing.TracerMultiPlanes(galaxies=[g0, g1, g2, g3, g4, g5],
                                                   image_plane_grid_stack=grid_stack)

            assert tracer.galaxies == [g0, g1, g4, g2, g3, g5]

        def test__galaxy_in_planes_lists__comes_in_lists_of_planes_in_redshift_order(self, grid_stack):
            g0 = g.Galaxy(redshift=0.5)
            g1 = g.Galaxy(redshift=0.5)

            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g0, g1], image_plane_grid_stack=grid_stack)

            assert tracer.galaxies_in_planes == [[g0, g1]]

            g2 = g.Galaxy(redshift=1.0)
            g3 = g.Galaxy(redshift=1.0)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1], source_galaxies=[g2, g3],
                                                         image_plane_grid_stack=grid_stack)

            assert tracer.galaxies_in_planes == [[g0, g1], [g2, g3]]

            g4 = g.Galaxy(redshift=0.75)
            g5 = g.Galaxy(redshift=1.5)

            tracer = ray_tracing.TracerMultiPlanes(galaxies=[g0, g1, g2, g3, g4, g5],
                                                   image_plane_grid_stack=grid_stack)

            assert tracer.galaxies_in_planes == [[g0, g1], [g4], [g2, g3], [g5]]

    class TestGridAtRedshift:

        def test__lens_z05_source_z01_redshifts__match_planes_redshifts__gives_same_grids(self, grid_stack):
            g0 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0))
            g1 = g.Galaxy(redshift=1.0)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                         image_plane_grid_stack=grid_stack)

            traced_grid = tracer.grid_at_redshift_from_image_plane_grid_and_redshift(
                image_plane_grid=grid_stack.regular, redshift=0.5)

            assert (traced_grid == tracer.image_plane.grid_stack.regular).all()

            traced_grid = tracer.grid_at_redshift_from_image_plane_grid_and_redshift(
                image_plane_grid=grid_stack.regular, redshift=1.0)

            assert (traced_grid == tracer.source_plane.grid_stack.regular).all()

        def test__same_as_above__input_grid_is_not_grid_stack(self, grid_stack):
            g0 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0))
            g1 = g.Galaxy(redshift=1.0)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                         image_plane_grid_stack=grid_stack)

            traced_grid = tracer.grid_at_redshift_from_image_plane_grid_and_redshift(
                image_plane_grid=np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]), redshift=0.5)

            assert (traced_grid == tracer.image_plane.grid_stack.regular[0]).all()

        def test__same_as_above_but_for_multi_tracer(self, grid_stack):
            g0 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0))
            g1 = g.Galaxy(redshift=0.75, mass_profile=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0))
            g2 = g.Galaxy(redshift=1.5, mass_profile=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=3.0))
            g3 = g.Galaxy(redshift=1.0, mass_profile=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=4.0))
            g4 = g.Galaxy(redshift=2.0)

            tracer = ray_tracing.TracerMultiPlanes(galaxies=[g0, g1, g2, g3, g4], image_plane_grid_stack=grid_stack)

            traced_grid = tracer.grid_at_redshift_from_image_plane_grid_and_redshift(
                image_plane_grid=grid_stack.regular, redshift=0.5)

            assert (traced_grid == tracer.planes[0].grid_stack.regular).all()

            traced_grid = tracer.grid_at_redshift_from_image_plane_grid_and_redshift(
                image_plane_grid=grid_stack.regular, redshift=0.75)

            assert (traced_grid == tracer.planes[1].grid_stack.regular).all()

            traced_grid = tracer.grid_at_redshift_from_image_plane_grid_and_redshift(
                image_plane_grid=grid_stack.regular, redshift=1.0)

            assert (traced_grid == tracer.planes[2].grid_stack.regular).all()

            traced_grid = tracer.grid_at_redshift_from_image_plane_grid_and_redshift(
                image_plane_grid=grid_stack.regular, redshift=1.5)

            assert (traced_grid == tracer.planes[3].grid_stack.regular).all()

            traced_grid = tracer.grid_at_redshift_from_image_plane_grid_and_redshift(
                image_plane_grid=grid_stack.regular, redshift=2.0)

            assert (traced_grid == tracer.planes[4].grid_stack.regular).all()

        def test__input_redshift_between_two_planes__performs_ray_tracing_calculation_correctly(self, grid_stack):

            g0 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0))
            g1 = g.Galaxy(redshift=0.75, mass_profile=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0))
            g2 = g.Galaxy(redshift=2.0)

            tracer = ray_tracing.TracerMultiPlanes(galaxies=[g0, g1, g2], image_plane_grid_stack=grid_stack)

            traced_grid = tracer.grid_at_redshift_from_image_plane_grid_and_redshift(
                image_plane_grid=grid_stack.regular, redshift=0.6)

            scaling_factor = cosmology_util.scaling_factor_between_redshifts_from_redshifts_and_cosmology(
                redshift_0=0.5, redshift_1=0.6, redshift_final=2.0, cosmology=tracer.cosmology)

            deflections_stack = lens_util.scaled_deflections_stack_from_plane_and_scaling_factor(
                plane=tracer.planes[0], scaling_factor=scaling_factor)

            traced_grid_stack_manual = lens_util.grid_stack_from_deflections_stack(grid_stack=grid_stack,
                                                                                   deflections_stack=deflections_stack)

            assert (traced_grid_stack_manual.regular == traced_grid).all()

        def test__input_redshift_between_two_planes__two_planes_between_earth_and_input_redshift(self, grid_stack):

            g0 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0))
            g1 = g.Galaxy(redshift=0.75, mass_profile=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0))
            g2 = g.Galaxy(redshift=2.0)

            tracer = ray_tracing.TracerMultiPlanes(galaxies=[g0, g1, g2], image_plane_grid_stack=grid_stack)

            traced_grid = tracer.grid_at_redshift_from_image_plane_grid_and_redshift(
                image_plane_grid=grid_stack.regular, redshift=1.9)

            # First loop, Plane index = 0 #

            scaling_factor = cosmology_util.scaling_factor_between_redshifts_from_redshifts_and_cosmology(
                redshift_0=0.5, redshift_1=1.9, redshift_final=2.0, cosmology=tracer.cosmology)

            deflections_stack = lens_util.scaled_deflections_stack_from_plane_and_scaling_factor(
                plane=tracer.planes[0], scaling_factor=scaling_factor)

            traced_grid_stack_manual = lens_util.grid_stack_from_deflections_stack(
                grid_stack=grid_stack, deflections_stack=deflections_stack)

            # Second loop, Plane index = 1 #

            scaling_factor = cosmology_util.scaling_factor_between_redshifts_from_redshifts_and_cosmology(
                redshift_0=0.75, redshift_1=1.9, redshift_final=2.0, cosmology=tracer.cosmology)

            deflections_stack = lens_util.scaled_deflections_stack_from_plane_and_scaling_factor(
                plane=tracer.planes[1], scaling_factor=scaling_factor)

            traced_grid_stack_manual = lens_util.grid_stack_from_deflections_stack(
                grid_stack=traced_grid_stack_manual, deflections_stack=deflections_stack)

            assert (traced_grid_stack_manual.regular == traced_grid).all()

        def test__input_redshift_before_first_plane__returns_image_plane(self, grid_stack):
            g0 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0))
            g1 = g.Galaxy(redshift=0.75, mass_profile=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0))

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                         image_plane_grid_stack=grid_stack)

            traced_grid = tracer.grid_at_redshift_from_image_plane_grid_and_redshift(
                image_plane_grid=grid_stack.regular, redshift=0.3)

            assert (traced_grid == grid_stack.regular).all()

    class TestEinsteinRadiusAndMass:

        def test__x2_lens_galaxies__values_are_sum_of_each_galaxy(self, grid_stack):

            g0 = g.Galaxy(redshift=1.0, mass=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(redshift=1.0, mass=mp.SphericalIsothermal(einstein_radius=2.0))

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1],
                                                         source_galaxies=[g.Galaxy(redshift=2.0)],
                                                         image_plane_grid_stack=grid_stack,
                                                         cosmology=cosmo.Planck15)

            g0_einstein_radius = g0.einstein_radius_in_units(unit_length='arcsec')
            g1_einstein_radius = g1.einstein_radius_in_units(unit_length='arcsec')

            assert tracer.einstein_radius_of_plane_in_units(i=0, unit_length='arcsec') == g0_einstein_radius + g1_einstein_radius
            assert tracer.einstein_radius_of_plane_in_units(i=1, unit_length='arcsec') is None

            # g0_mass = g0.einstein_mass_in_units(unit_mass='angular')
            # g1_mass = g1.einstein_mass_in_units(unit_mass='angular')
            # assert tracer.einstein_mass_of_plane_in_units(i=0, unit_mass='angular') == g0_mass + g1_mass
            # assert tracer.einstein_mass_of_plane_in_units(i=1, unit_mass='angular') is None

            g0_einstein_radius = g0.einstein_radius_in_units(unit_length='kpc')
            g1_einstein_radius = g1.einstein_radius_in_units(unit_length='kpc')
            assert tracer.einstein_radius_of_plane_in_units(i=0, unit_length='kpc') == g0_einstein_radius + g1_einstein_radius
            assert tracer.einstein_radius_of_plane_in_units(i=1, unit_length='kpc') is None

            g0_mass = g0.einstein_mass_in_units(unit_mass='solMass', redshift_source=2.0)
            g1_mass = g1.einstein_mass_in_units(unit_mass='solMass', redshift_source=2.0)
            assert tracer.einstein_mass_between_planes_in_units(i=0, j=1, unit_mass='solMass') == g0_mass + g1_mass
            assert tracer.einstein_mass_between_planes_in_units(i=1, j=1, unit_mass='solMass') is None

        def test__same_as_above__include_shear__does_not_impact_calculations(self, grid_stack):

            g0 = g.Galaxy(redshift=1.0, mass=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(redshift=1.0, mass=mp.SphericalIsothermal(einstein_radius=2.0), shear=mp.ExternalShear())

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1],
                                                         source_galaxies=[g.Galaxy(redshift=2.0)],
                                                         image_plane_grid_stack=grid_stack,
                                                         cosmology=cosmo.Planck15)

            g0_einstein_radius = g0.einstein_radius_in_units(unit_length='arcsec')
            g1_einstein_radius = g1.einstein_radius_in_units(unit_length='arcsec')

            assert tracer.einstein_radius_of_plane_in_units(i=0, unit_length='arcsec') == \
                   g0_einstein_radius + g1_einstein_radius
            assert tracer.einstein_radius_of_image_plane_in_units(unit_length='arcsec') == \
                   g0_einstein_radius + g1_einstein_radius
            assert tracer.einstein_radius_of_plane_in_units(i=1, unit_length='arcsec') is None

            # g0_mass = g0.einstein_mass_in_units(unit_mass='angular')
            # g1_mass = g1.einstein_mass_in_units(unit_mass='angular')
            # assert tracer.einstein_mass_of_plane_in_units(i=0, unit_mass='angular') == g0_mass + g1_mass
            # assert tracer.einstein_mass_of_plane_in_units(i=1, unit_mass='angular') is None

            g0_einstein_radius = g0.einstein_radius_in_units(unit_length='kpc')
            g1_einstein_radius = g1.einstein_radius_in_units(unit_length='kpc')
            assert tracer.einstein_radius_of_plane_in_units(i=0, unit_length='kpc') == \
                   g0_einstein_radius + g1_einstein_radius
            assert tracer.einstein_radius_of_image_plane_in_units(unit_length='kpc') == \
                   g0_einstein_radius + g1_einstein_radius
            assert tracer.einstein_radius_of_plane_in_units(i=1, unit_length='kpc') is None

            g0_mass = g0.einstein_mass_in_units(unit_mass='solMass', redshift_source=2.0)
            g1_mass = g1.einstein_mass_in_units(unit_mass='solMass', redshift_source=2.0)
            assert tracer.einstein_mass_between_planes_in_units(i=0, j=1, unit_mass='solMass') == g0_mass + g1_mass
            assert tracer.einstein_mass_between_planes_in_units(i=1, j=1, unit_mass='solMass') is None
            assert tracer.einstein_mass_between_image_and_source_plane_in_units(unit_mass='solMass') == \
                   g0_mass + g1_mass


@pytest.fixture(name='mask')
def make_mask():
    return msk.Mask(array=np.array([[True, True, True, True],
                                    [True, False, False, True],
                                    [True, False, False, True],
                                    [True, True, True, True]]), pixel_scale=1.0)


@pytest.fixture(name='blurring_mask')
def make_blurring_mask():
    return msk.Mask(array=np.array([[False, False, False, False],
                                    [False, True, True, False],
                                    [False, True, True, False],
                                    [False, False, False, False]]), pixel_scale=1.0)

@pytest.fixture(name='convolver_blur')
def make_convolver_blur(mask, blurring_mask):
    psf = np.array([[1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0]])
    return convolution.ConvolverImage(mask=mask, blurring_mask=blurring_mask, psf=psf)


class TestAbstractTracerData(object):
    
    class TestBlurredImages:

        def test__blurred_image_plane_image_1d_of_planes(self, grid_stack, convolver_blur):

            g0 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=2.0))

            plane_0 = pl.AbstractDataPlane(redshift=0.5, galaxies=[g0], grid_stack=grid_stack, compute_deflections=False,
                                         border=None)
            plane_1 = pl.AbstractDataPlane(redshift=0.5, galaxies=[g1], grid_stack=grid_stack, compute_deflections=False,
                                         border=None)
            plane_2 = pl.AbstractDataPlane(redshift=1.0, galaxies=[g.Galaxy(redshift=0.5)], grid_stack=grid_stack,
                                         compute_deflections=False, border=None)

            blurred_image_1d_0 = plane_0.blurred_profile_image_plane_image_1d_from_convolver_image(convolver_image=convolver_blur)
            blurred_image_1d_1 = plane_1.blurred_profile_image_plane_image_1d_from_convolver_image(convolver_image=convolver_blur)

            tracer = ray_tracing.AbstractTracerData(planes=[plane_0, plane_1, plane_2], cosmology=cosmo.Planck15)

            blurred_image_plane_images_1d = \
                tracer.blurred_profile_image_plane_image_1d_of_planes_from_convolver_image(convolver_image=convolver_blur)

            assert (blurred_image_plane_images_1d[0] == blurred_image_1d_0).all()
            assert (blurred_image_plane_images_1d[1] == blurred_image_1d_1).all()
            assert (blurred_image_plane_images_1d[2] == np.zeros(blurred_image_1d_0.shape)).all()

            blurred_image_plane_image_1d = \
                tracer.blurred_profile_image_plane_image_1d_from_convolver_image(convolver_image=convolver_blur)

            assert blurred_image_plane_image_1d == pytest.approx(blurred_image_1d_0 + blurred_image_1d_1, 1.0e-4)

            blurred_image_plane_images = \
                tracer.blurred_profile_image_plane_image_2d_of_planes_from_convolver_image(convolver_image=convolver_blur)

            blurred_image_0 = grid_stack.scaled_array_2d_from_array_1d(array_1d=blurred_image_1d_0)
            blurred_image_1 = grid_stack.scaled_array_2d_from_array_1d(array_1d=blurred_image_1d_1)

            assert (blurred_image_plane_images[0] == blurred_image_0).all()
            assert (blurred_image_plane_images[1] == blurred_image_1).all()
            #    assert blurred_image_1ds[2] == None

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                         image_plane_grid_stack=grid_stack, cosmology=cosmo.Planck15)

            blurred_image_plane_images_1ds = \
                tracer.blurred_profile_image_plane_image_1d_of_planes_from_convolver_image(convolver_image=convolver_blur)

            assert (blurred_image_plane_images_1ds[0] == blurred_image_1d_0).all()
            assert (blurred_image_plane_images_1ds[1] == blurred_image_1d_1).all()

    class TestUnmaskedBlurredImages:

        def test__unmasked_images_of_tracer_planes_and_galaxies(self):

            psf = ccd.PSF(array=(np.array([[0.0, 3.0, 0.0],
                                          [0.0, 1.0, 2.0],
                                          [0.0, 0.0, 0.0]])), pixel_scale=1.0)

            mask = msk.Mask(array=np.array([[True, True, True],
                                            [True, False, True],
                                            [True, True, True]]), pixel_scale=1.0)

            padded_grid_stack = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                                        sub_grid_size=1,
                                                                                                        psf_shape=(
                                                                                                        3, 3))

            g0 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=0.1))
            g1 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=0.2))
            g2 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=0.3))
            g3 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=0.4))

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1], source_galaxies=[g2, g3],
                                                         image_plane_grid_stack=padded_grid_stack)

            manual_blurred_image_0 = tracer.image_plane.profile_image_plane_image_1d_of_galaxies[0]
            manual_blurred_image_0 = padded_grid_stack.regular.map_to_2d_keep_padded(
                padded_array_1d=manual_blurred_image_0)
            manual_blurred_image_0 = psf.convolve(array=manual_blurred_image_0)

            manual_blurred_image_1 = tracer.image_plane.profile_image_plane_image_1d_of_galaxies[1]
            manual_blurred_image_1 = padded_grid_stack.regular.map_to_2d_keep_padded(
                padded_array_1d=manual_blurred_image_1)
            manual_blurred_image_1 = psf.convolve(array=manual_blurred_image_1)

            manual_blurred_image_2 = tracer.source_plane.profile_image_plane_image_1d_of_galaxies[0]
            manual_blurred_image_2 = padded_grid_stack.regular.map_to_2d_keep_padded(
                padded_array_1d=manual_blurred_image_2)
            manual_blurred_image_2 = psf.convolve(array=manual_blurred_image_2)

            manual_blurred_image_3 = tracer.source_plane.profile_image_plane_image_1d_of_galaxies[1]
            manual_blurred_image_3 = padded_grid_stack.regular.map_to_2d_keep_padded(
                padded_array_1d=manual_blurred_image_3)
            manual_blurred_image_3 = psf.convolve(array=manual_blurred_image_3)

            unmasked_blurred_image = tracer.unmasked_blurred_profile_image_plane_image_from_psf(psf=psf)

            assert unmasked_blurred_image == \
                    pytest.approx(manual_blurred_image_0[1:4, 1:4] + manual_blurred_image_1[1:4, 1:4] +
                                  manual_blurred_image_2[1:4, 1:4] + manual_blurred_image_3[1:4, 1:4], 1.0e-4)

            unmasked_blurred_image_of_planes = \
                tracer.unmasked_blurred_profile_image_plane_image_of_planes_from_psf(psf=psf)

            assert unmasked_blurred_image_of_planes[0] == \
                    pytest.approx(manual_blurred_image_0[1:4, 1:4] + manual_blurred_image_1[1:4, 1:4], 1.0e-4)
            assert unmasked_blurred_image_of_planes[1] == \
                    pytest.approx(manual_blurred_image_2[1:4, 1:4] + manual_blurred_image_3[1:4, 1:4], 1.0e-4)

            unmasked_blurred_image_of_planes_and_galaxies = \
                tracer.unmasked_blurred_profile_image_plane_image_of_plane_and_galaxies_from_psf(psf=psf)

            assert (unmasked_blurred_image_of_planes_and_galaxies[0][0] == manual_blurred_image_0[1:4, 1:4]).all()
            assert (unmasked_blurred_image_of_planes_and_galaxies[0][1] == manual_blurred_image_1[1:4, 1:4]).all()
            assert (unmasked_blurred_image_of_planes_and_galaxies[1][0] == manual_blurred_image_2[1:4, 1:4]).all()
            assert (unmasked_blurred_image_of_planes_and_galaxies[1][1] == manual_blurred_image_3[1:4, 1:4]).all()

    class TestHyperNoiseMap:

        def test__hyper_noise_maps_of_planes(self, grid_stack):

            noise_map_1d = np.array([5.0, 3.0, 1.0])

            hyper_model_image_1d = np.array([2.0, 4.0, 10.0])
            hyper_galaxy_image_1d = np.array([1.0, 5.0, 8.0])

            hyper_galaxy_0 = g.HyperGalaxy(contribution_factor=5.0)
            hyper_galaxy_1 = g.HyperGalaxy(contribution_factor=10.0)

            galaxy_0 = g.Galaxy(redshift=0.5, hyper_galaxy=hyper_galaxy_0, hyper_model_image_1d=hyper_model_image_1d,
                                hyper_galaxy_image_1d=hyper_galaxy_image_1d, hyper_minimum_value=0.0)

            galaxy_1 = g.Galaxy(redshift=0.5, hyper_galaxy=hyper_galaxy_1, hyper_model_image_1d=hyper_model_image_1d,
                                hyper_galaxy_image_1d=hyper_galaxy_image_1d, hyper_minimum_value=0.0)

            plane_0 = pl.AbstractDataPlane(redshift=0.5, galaxies=[galaxy_0], grid_stack=None, compute_deflections=False,
                                         border=None)
            plane_1 = pl.AbstractDataPlane(redshift=0.5, galaxies=[galaxy_1], grid_stack=None, compute_deflections=False,
                                         border=None)
            plane_2 = pl.AbstractDataPlane(redshift=1.0, galaxies=[g.Galaxy(redshift=0.5)], grid_stack=None,
                                         compute_deflections=False, border=None)

            hyper_noise_map_1d_0 = plane_0.hyper_noise_map_1d_from_noise_map_1d(noise_map_1d=noise_map_1d)
            hyper_noise_map_1d_1 = plane_1.hyper_noise_map_1d_from_noise_map_1d(noise_map_1d=noise_map_1d)

            tracer = ray_tracing.AbstractTracerData(planes=[plane_0, plane_1, plane_2], cosmology=cosmo.Planck15)

            hyper_noise_maps_1d = tracer.hyper_noise_maps_1d_of_planes_from_noise_map_1d(noise_map_1d=noise_map_1d)

            assert (hyper_noise_maps_1d[0] == hyper_noise_map_1d_0).all()
            assert (hyper_noise_maps_1d[1] == hyper_noise_map_1d_1).all()
            assert hyper_noise_maps_1d[2] == None

            hyper_noise_map_1d = tracer.hyper_noise_map_1d_from_noise_map_1d(noise_map_1d=noise_map_1d)

            assert (hyper_noise_map_1d == hyper_noise_map_1d_0 + hyper_noise_map_1d_1).all()

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_0], source_galaxies=[galaxy_1],
                                                         image_plane_grid_stack=grid_stack, cosmology=cosmo.Planck15)

            hyper_noise_maps_1d = tracer.hyper_noise_maps_1d_of_planes_from_noise_map_1d(noise_map_1d=noise_map_1d)

            assert (hyper_noise_maps_1d[0] == hyper_noise_map_1d_0).all()
            assert (hyper_noise_maps_1d[1] == hyper_noise_map_1d_1).all()


class TestTracerImagePlane(object):

    class TestProfileImagePlaneImage:

        def test__1_plane__single_plane_tracer(self, grid_stack):
            g0 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=2.0))
            g2 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=3.0))

            image_plane = pl.Plane(galaxies=[g0, g1, g2], grid_stack=grid_stack, compute_deflections=True)

            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g0, g1, g2], image_plane_grid_stack=grid_stack)

            assert (tracer.profile_image_plane_image_1d == image_plane.profile_image_plane_image_1d).all()

            image_plane_image_2d = grid_stack.regular.scaled_array_2d_from_array_1d(image_plane.profile_image_plane_image_1d)
            assert image_plane_image_2d.shape == (3, 4)
            assert (image_plane_image_2d == tracer.profile_image_plane_image_2d).all()

    class TestProfileImagePlaneBlurringImages:

        def test__1_plane__single_plane_tracer(self, grid_stack):
            g0 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=2.0))
            g2 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=3.0))

            image_plane = pl.Plane(galaxies=[g0, g1, g2], grid_stack=grid_stack, compute_deflections=True)

            tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g0, g1, g2], image_plane_grid_stack=grid_stack)

            assert (tracer.profile_image_plane_blurring_image_1d == image_plane.profile_image_plane_blurring_image_1d).all()


class TestTracerImageSourcePlanes(object):

    class TestSetup:

        def test__no_galaxy__image_and_source_planes_setup__same_coordinates(self, grid_stack, galaxy_non):
            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_non], source_galaxies=[galaxy_non],
                                                         image_plane_grid_stack=grid_stack)

            assert tracer.image_plane.grid_stack.regular[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.image_plane.grid_stack.sub[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.image_plane.grid_stack.sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert tracer.image_plane.grid_stack.sub[2] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.image_plane.grid_stack.sub[3] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

            assert tracer.image_plane.deflections_stack.regular[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflections_stack.sub[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflections_stack.sub[1] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflections_stack.sub[2] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflections_stack.sub[3] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflections_stack.blurring[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)

            assert tracer.source_plane.grid_stack.regular[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.source_plane.grid_stack.sub[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.source_plane.grid_stack.sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert tracer.source_plane.grid_stack.sub[2] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.source_plane.grid_stack.sub[3] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

        def test__sis_lens__image_sub_and_blurring_grid_stack_on_planes_setup(self, grid_stack, galaxy_mass):
            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_mass], source_galaxies=[galaxy_mass],
                                                         image_plane_grid_stack=grid_stack)

            assert tracer.image_plane.grid_stack.regular[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.image_plane.grid_stack.sub[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.image_plane.grid_stack.sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert tracer.image_plane.grid_stack.sub[2] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.image_plane.grid_stack.sub[3] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert tracer.image_plane.grid_stack.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

            assert tracer.image_plane.deflections_stack.regular[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert tracer.image_plane.deflections_stack.sub[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert tracer.image_plane.deflections_stack.sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflections_stack.sub[2] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert tracer.image_plane.deflections_stack.sub[3] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflections_stack.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

            assert tracer.source_plane.grid_stack.regular[0] == pytest.approx(np.array([1.0 - 0.707, 1.0 - 0.707]),
                                                                              1e-3)
            assert tracer.source_plane.grid_stack.sub[0] == pytest.approx(np.array([1.0 - 0.707, 1.0 - 0.707]), 1e-3)
            assert tracer.source_plane.grid_stack.sub[1] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.source_plane.grid_stack.sub[2] == pytest.approx(np.array([1.0 - 0.707, 1.0 - 0.707]), 1e-3)
            assert tracer.source_plane.grid_stack.sub[3] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.source_plane.grid_stack.blurring[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)

        def test__same_as_above_but_2_sis_lenses__deflections_double(self, grid_stack, galaxy_mass):
            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_mass, galaxy_mass],
                                                         source_galaxies=[galaxy_mass],
                                                         image_plane_grid_stack=grid_stack)

            assert tracer.image_plane.grid_stack.regular[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.image_plane.grid_stack.sub[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.image_plane.grid_stack.sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert tracer.image_plane.grid_stack.sub[2] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.image_plane.grid_stack.sub[3] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert tracer.image_plane.grid_stack.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

            assert tracer.image_plane.deflections_stack.regular[0] == pytest.approx(np.array([2.0 * 0.707, 2.0 * 0.707]),
                                                                                   1e-3)
            assert tracer.image_plane.deflections_stack.sub[0] == pytest.approx(np.array([2.0 * 0.707, 2.0 * 0.707]),
                                                                               1e-3)
            assert tracer.image_plane.deflections_stack.sub[1] == pytest.approx(np.array([2.0 * 1.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflections_stack.sub[2] == pytest.approx(np.array([2.0 * 0.707, 2.0 * 0.707]),
                                                                               1e-3)
            assert tracer.image_plane.deflections_stack.sub[3] == pytest.approx(np.array([2.0 * 1.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflections_stack.blurring[0] == pytest.approx(np.array([2.0 * 1.0, 0.0]), 1e-3)

            assert tracer.source_plane.grid_stack.regular[0] == pytest.approx(
                np.array([1.0 - 2.0 * 0.707, 1.0 - 2.0 * 0.707]),
                1e-3)
            assert tracer.source_plane.grid_stack.sub[0] == pytest.approx(
                np.array([1.0 - 2.0 * 0.707, 1.0 - 2.0 * 0.707]),
                1e-3)
            assert tracer.source_plane.grid_stack.sub[1] == pytest.approx(np.array([-1.0, 0.0]), 1e-3)
            assert tracer.source_plane.grid_stack.sub[2] == pytest.approx(
                np.array([1.0 - 2.0 * 0.707, 1.0 - 2.0 * 0.707]),
                1e-3)
            assert tracer.source_plane.grid_stack.sub[3] == pytest.approx(np.array([-1.0, 0.0]), 1e-3)
            assert tracer.source_plane.grid_stack.blurring[0] == pytest.approx(np.array([-1.0, 0.0]), 1e-3)

        def test__grid_attributes_passed(self, grid_stack, galaxy_non):
            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_non], source_galaxies=[galaxy_non],
                                                         image_plane_grid_stack=grid_stack)

            assert (tracer.image_plane.grid_stack.regular.mask == grid_stack.regular.mask).all()
            assert (tracer.image_plane.grid_stack.sub.mask == grid_stack.sub.mask).all()
            assert (tracer.source_plane.grid_stack.regular.mask == grid_stack.regular.mask).all()
            assert (tracer.source_plane.grid_stack.sub.mask == grid_stack.sub.mask).all()

    class TestProfileImagePlaneImage:

        def test__galaxy_light__no_mass__image_sum_of_image_and_source_plane(self, grid_stack):
            g0 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=2.0))

            image_plane = pl.Plane(galaxies=[g0], grid_stack=grid_stack, compute_deflections=True)
            source_plane = pl.Plane(galaxies=[g1], grid_stack=grid_stack, compute_deflections=False)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                         image_plane_grid_stack=grid_stack)

            image_plane_image_1d = image_plane.profile_image_plane_image_1d + source_plane.profile_image_plane_image_1d

            assert (image_plane_image_1d == tracer.profile_image_plane_image_1d).all()

            image_plane_image_2d = grid_stack.regular.scaled_array_2d_from_array_1d(
                image_plane.profile_image_plane_image_1d) + grid_stack.regular.scaled_array_2d_from_array_1d(
                source_plane.profile_image_plane_image_1d)
            assert image_plane_image_2d.shape == (3, 4)
            assert (image_plane_image_2d == tracer.profile_image_plane_image_2d).all()

        def test__galaxy_light_mass_sis__source_plane_image_includes_deflections(self, grid_stack):
            g0 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=1.0),
                          mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=2.0))

            image_plane = pl.Plane(galaxies=[g0], grid_stack=grid_stack, compute_deflections=True)

            deflections_grid = galaxy_util.deflections_of_galaxies_from_grid_stack(grid_stack, galaxies=[g0])
            source_grid_stack = lens_util.traced_collection_for_deflections(grid_stack, deflections_grid)
            source_plane = pl.Plane(galaxies=[g1], grid_stack=source_grid_stack, compute_deflections=False)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                         image_plane_grid_stack=grid_stack)

            image_plane_image_1d = image_plane.profile_image_plane_image_1d + source_plane.profile_image_plane_image_1d
            assert (image_plane_image_1d == tracer.profile_image_plane_image_1d).all()

            image_plane_image_2d = grid_stack.regular.scaled_array_2d_from_array_1d(
                image_plane.profile_image_plane_image_1d) + grid_stack.regular.scaled_array_2d_from_array_1d(
                source_plane.profile_image_plane_image_1d)
            assert image_plane_image_2d.shape == (3, 4)
            assert (image_plane_image_2d == tracer.profile_image_plane_image_2d).all()

        def test__image_plane_image__compare_to_galaxy_images(self, grid_stack):
            g0 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=2.0))
            g2 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=3.0))

            g0_image = galaxy_util.intensities_of_galaxies_from_grid(grid_stack.sub, galaxies=[g0])
            g1_image = galaxy_util.intensities_of_galaxies_from_grid(grid_stack.sub, galaxies=[g1])
            g2_image = galaxy_util.intensities_of_galaxies_from_grid(grid_stack.sub, galaxies=[g2])

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1], source_galaxies=[g2],
                                                         image_plane_grid_stack=grid_stack)

            assert tracer.profile_image_plane_image_1d == pytest.approx(g0_image + g1_image + g2_image, 1.0e-4)

            assert tracer.profile_image_plane_image_2d == pytest.approx(
                grid_stack.regular.scaled_array_2d_from_array_1d(g0_image) +
                grid_stack.regular.scaled_array_2d_from_array_1d(g1_image) +
                grid_stack.regular.scaled_array_2d_from_array_1d(g2_image),
                1.0e-4)

        def test__2_planes__returns_image_plane_image_of_each_plane(self, grid_stack):
            g0 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=1.0),
                          mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))

            image_plane = pl.Plane(galaxies=[g0], grid_stack=grid_stack, compute_deflections=True)
            source_plane_grid_stack = image_plane.trace_grid_stack_to_next_plane()
            source_plane = pl.Plane(galaxies=[g0], grid_stack=source_plane_grid_stack, compute_deflections=False)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g0],
                                                         image_plane_grid_stack=grid_stack)

            assert (tracer.profile_image_plane_image_1d == image_plane.profile_image_plane_image_1d +
                    source_plane.profile_image_plane_image_1d).all()

            image_plane_image_2d = grid_stack.regular.scaled_array_2d_from_array_1d(
                image_plane.profile_image_plane_image_1d) + grid_stack.regular.scaled_array_2d_from_array_1d(
                source_plane.profile_image_plane_image_1d)
            assert image_plane_image_2d.shape == (3, 4)
            assert (image_plane_image_2d == tracer.profile_image_plane_image_2d).all()

        def test__padded_2d_image_from_plane__mapped_correctly(self, padded_grid_stack, galaxy_light, galaxy_mass):
            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_light, galaxy_mass],
                                                         source_galaxies=[galaxy_light],
                                                         image_plane_grid_stack=padded_grid_stack)

            image_plane_image_2d = padded_grid_stack.regular.scaled_array_2d_from_array_1d(
                tracer.image_plane.profile_image_plane_image_1d) + padded_grid_stack.regular.scaled_array_2d_from_array_1d(
                tracer.source_plane.profile_image_plane_image_1d)

            assert image_plane_image_2d.shape == (1, 2)
            assert (image_plane_image_2d == tracer.profile_image_plane_image_2d).all()

        def test__padded_2d_image_for_simulation__mapped_correctly_not_trimmed(self, padded_grid_stack, galaxy_light,
                                                                               galaxy_mass):
            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_light, galaxy_mass],
                                                         source_galaxies=[galaxy_light],
                                                         image_plane_grid_stack=padded_grid_stack)

            image_plane_image_2d = padded_grid_stack.regular.map_to_2d_keep_padded(
                tracer.image_plane.profile_image_plane_image_1d) + padded_grid_stack.regular.map_to_2d_keep_padded(
                tracer.source_plane.profile_image_plane_image_1d)

            assert image_plane_image_2d.shape == (3, 4)
            assert (image_plane_image_2d == tracer.profile_image_plane_image_2d_for_simulation).all()

            assert (tracer.profile_image_plane_image_2d_for_simulation == tracer.profile_image_plane_image_2d_for_simulation).all()

    class TestProfileImagePlaneBlurringImages:

        def test__galaxy_light__no_mass__image_sum_of_image_and_source_plane(self, grid_stack):
            g0 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=2.0))

            image_plane = pl.Plane(galaxies=[g0], grid_stack=grid_stack, compute_deflections=True)
            source_plane = pl.Plane(galaxies=[g1], grid_stack=grid_stack, compute_deflections=False)

            image_plane_blurring_image = (image_plane.profile_image_plane_blurring_image_1d +
                                          source_plane.profile_image_plane_blurring_image_1d)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                         image_plane_grid_stack=grid_stack)

            assert (image_plane_blurring_image == tracer.profile_image_plane_blurring_image_1d).all()

        def test__galaxy_light_mass_sis__source_plane_image_includes_deflections(self, grid_stack):
            g0 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=1.0),
                          mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(redshift=0.5, light_profile=lp.EllipticalSersic(intensity=2.0))

            image_plane = pl.Plane(galaxies=[g0], grid_stack=grid_stack, compute_deflections=True)

            deflection_grid_stack = galaxy_util.deflections_of_galaxies_from_grid_stack(grid_stack, galaxies=[g0])
            source_grid_stack = lens_util.traced_collection_for_deflections(grid_stack, deflection_grid_stack)
            source_plane = pl.Plane(galaxies=[g1], grid_stack=source_grid_stack, compute_deflections=False)

            image_plane_blurring_image = (image_plane.profile_image_plane_blurring_image_1d +
                                          source_plane.profile_image_plane_blurring_image_1d)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                         image_plane_grid_stack=grid_stack)

            assert (image_plane_blurring_image == tracer.profile_image_plane_blurring_image_1d).all()

    class TestImagePlanePixGrid:

        def test__galaxies_have_no_pixelization__no_pix_grid_added(self):
            mask = msk.Mask(np.array([[False, False, False],
                                      [False, False, False],
                                      [False, False, False]]), pixel_scale=1.0)

            grid_stack = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask, sub_grid_size=1,
                                                                                          psf_shape=(1, 1))

            galaxy = g.Galaxy(redshift=0.5)

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy], source_galaxies=[galaxy],
                                                         image_plane_grid_stack=grid_stack)

            assert (tracer.image_plane.grid_stack.pix == np.array([[0.0, 0.0]])).all()
            assert (tracer.source_plane.grid_stack.pix == np.array([[0.0, 0.0]])).all()

        def test__setup_pixelization__galaxies_have_other_pixelization__returns_normal_grids(self):
            mask = msk.Mask(np.array([[False, False, False],
                                      [False, False, False],
                                      [False, False, False]]), pixel_scale=1.0)

            grid_stack = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask, sub_grid_size=1,
                                                                                          psf_shape=(1, 1))

            galaxy = g.Galaxy(redshift=0.5, pixelization=pixelizations.Rectangular(shape=(3, 3)),
                              regularization=regularization.Constant())

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy], source_galaxies=[galaxy],
                                                         image_plane_grid_stack=grid_stack)

            assert (tracer.image_plane.grid_stack.pix == np.array([[0.0, 0.0]])).all()
            assert (tracer.source_plane.grid_stack.pix == np.array([[0.0, 0.0]])).all()

        def test__setup_pixelization__galaxy_has_pixelization__but_grid_is_padded_grid__returns_normal_grids(self):
            mask = msk.Mask(np.array([[False, False, False],
                                      [False, False, False],
                                      [False, True, False]]), pixel_scale=1.0)

            grid_stack = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                                 sub_grid_size=1,
                                                                                                 psf_shape=(1, 1))

            galaxy = g.Galaxy(redshift=0.5, pixelization=pixelizations.AdaptiveMagnification(shape=(3, 3)),
                              regularization=regularization.Constant())

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy], source_galaxies=[galaxy],
                                                         image_plane_grid_stack=grid_stack)

            assert (tracer.image_plane.grid_stack.pix == np.array([[0.0, 0.0]])).all()
            assert (tracer.source_plane.grid_stack.pix == np.array([[0.0, 0.0]])).all()

        def test__setup_pixelization__galaxy_has_pixelization__returns_grids_with_pix_grid(self):
            mask = msk.Mask(np.array([[False, False, False],
                                      [False, False, False],
                                      [False, True, False]]), pixel_scale=1.0)

            grid_stack = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask, sub_grid_size=1,
                                                                                          psf_shape=(1, 1))

            galaxy = g.Galaxy(redshift=0.5, pixelization=pixelizations.AdaptiveMagnification(shape=(3, 3)),
                              regularization=regularization.Constant())

            tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy], source_galaxies=[galaxy],
                                                         image_plane_grid_stack=grid_stack)

            assert (tracer.image_plane.grid_stack.regular == grid_stack.regular).all()
            assert (tracer.image_plane.grid_stack.sub == grid_stack.sub).all()
            assert (tracer.image_plane.grid_stack.blurring == grid_stack.blurring).all()
            assert (tracer.image_plane.grid_stack.pix == np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0],
                                                                   [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                                                   [-1.0, -1.0], [-1.0, 1.0]])).all()

            assert (tracer.source_plane.grid_stack.regular == grid_stack.regular).all()
            assert (tracer.source_plane.grid_stack.sub == grid_stack.sub).all()
            assert (tracer.source_plane.grid_stack.blurring == grid_stack.blurring).all()
            assert (tracer.source_plane.grid_stack.pix == np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0],
                                                                    [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                                                    [-1.0, -1.0], [-1.0, 1.0]])).all()


class TestMultiTracer(object):

    class TestCosmology:

        def test__3_planes__z01_z1__and_z2(self, grid_stack):

            g0 = g.Galaxy(redshift=0.1)
            g1 = g.Galaxy(redshift=1.0)
            g2 = g.Galaxy(redshift=2.0)

            tracer = ray_tracing.TracerMultiPlanes(galaxies=[g0, g1, g2], image_plane_grid_stack=grid_stack,
                                                   cosmology=cosmo.Planck15)

            assert tracer.arcsec_per_kpc_proper_of_plane(i=0) == pytest.approx(0.525060, 1e-5)
            assert tracer.kpc_per_arcsec_proper_of_plane(i=0) == pytest.approx(1.904544, 1e-5)

            assert tracer.angular_diameter_distance_of_plane_to_earth_in_units(i=0, unit_length='kpc') == \
                   pytest.approx(392840, 1e-5)
            assert tracer.angular_diameter_distance_between_planes_in_units(i=0, j=0, unit_length='kpc') == 0.0
            assert tracer.angular_diameter_distance_between_planes_in_units(i=0, j=1, unit_length='kpc') == \
                   pytest.approx(1481890.4, 1e-5)
            assert tracer.angular_diameter_distance_between_planes_in_units(i=0, j=2, unit_length='kpc') == \
                   pytest.approx(1626471, 1e-5)

            assert tracer.arcsec_per_kpc_proper_of_plane(i=1) == pytest.approx(0.1214785, 1e-5)
            assert tracer.kpc_per_arcsec_proper_of_plane(i=1) == pytest.approx(8.231907, 1e-5)

            assert tracer.angular_diameter_distance_of_plane_to_earth_in_units(i=1, unit_length='kpc') == \
                   pytest.approx(1697952, 1e-5)
            assert tracer.angular_diameter_distance_between_planes_in_units(i=1, j=0, unit_length='kpc') == \
                   pytest.approx(-2694346, 1e-5)
            assert tracer.angular_diameter_distance_between_planes_in_units(i=1, j=1, unit_length='kpc') == 0.0
            assert tracer.angular_diameter_distance_between_planes_in_units(i=1, j=2, unit_length='kpc') == \
                   pytest.approx(638544, 1e-5)

            assert tracer.arcsec_per_kpc_proper_of_plane(i=2) == pytest.approx(0.116500, 1e-5)
            assert tracer.kpc_per_arcsec_proper_of_plane(i=2) == pytest.approx(8.58368, 1e-5)

            assert tracer.angular_diameter_distance_of_plane_to_earth_in_units(i=2, unit_length='kpc') == \
                   pytest.approx(1770512, 1e-5)
            assert tracer.angular_diameter_distance_between_planes_in_units(i=2, j=0, unit_length='kpc') == \
                   pytest.approx(-4435831, 1e-5)
            assert tracer.angular_diameter_distance_between_planes_in_units(i=2, j=1, unit_length='kpc') == \
                   pytest.approx(-957816)
            assert tracer.angular_diameter_distance_between_planes_in_units(i=2, j=2, unit_length='kpc') == 0.0

            assert tracer.critical_surface_density_between_planes_in_units(
                i=0, j=1, unit_length='kpc', unit_mass='solMass') == pytest.approx(4.85e9, 1e-2)
            assert tracer.critical_surface_density_between_planes_in_units(
                i=0, j=1, unit_length='arcsec', unit_mass='solMass') == pytest.approx(17593241668, 1e-2)

            assert tracer.scaling_factor_between_planes(i=0, j=1) == pytest.approx(0.9500, 1e-4)
            assert tracer.scaling_factor_between_planes(i=0, j=2) == pytest.approx(1.0, 1e-4)
            assert tracer.scaling_factor_between_planes(i=1, j=2) == pytest.approx(1.0, 1e-4)

        def test__4_planes__z01_z1_z2_and_z3(self, grid_stack):

            g0 = g.Galaxy(redshift=0.1)
            g1 = g.Galaxy(redshift=1.0)
            g2 = g.Galaxy(redshift=2.0)
            g3 = g.Galaxy(redshift=3.0)

            tracer = ray_tracing.TracerMultiPlanes(galaxies=[g0, g1, g2, g3], image_plane_grid_stack=grid_stack,
                                                   cosmology=cosmo.Planck15)

            assert tracer.arcsec_per_kpc_proper_of_plane(i=0) == pytest.approx(0.525060, 1e-5)
            assert tracer.kpc_per_arcsec_proper_of_plane(i=0) == pytest.approx(1.904544, 1e-5)

            assert tracer.angular_diameter_distance_of_plane_to_earth_in_units(i=0, unit_length='kpc') == \
                   pytest.approx(392840, 1e-5)
            assert tracer.angular_diameter_distance_between_planes_in_units(i=0, j=0, unit_length='kpc') == 0.0
            assert tracer.angular_diameter_distance_between_planes_in_units(i=0, j=1, unit_length='kpc') == \
                   pytest.approx(1481890.4, 1e-5)
            assert tracer.angular_diameter_distance_between_planes_in_units(i=0, j=2, unit_length='kpc') == \
                   pytest.approx(1626471, 1e-5)
            assert tracer.angular_diameter_distance_between_planes_in_units(i=0, j=3, unit_length='kpc') == \
                   pytest.approx(1519417, 1e-5)

            assert tracer.arcsec_per_kpc_proper_of_plane(i=1) == pytest.approx(0.1214785, 1e-5)
            assert tracer.kpc_per_arcsec_proper_of_plane(i=1) == pytest.approx(8.231907, 1e-5)

            assert tracer.angular_diameter_distance_of_plane_to_earth_in_units(i=1, unit_length='kpc') == \
                   pytest.approx(1697952, 1e-5)
            assert tracer.angular_diameter_distance_between_planes_in_units(i=1, j=0, unit_length='kpc') == \
                   pytest.approx(-2694346, 1e-5)
            assert tracer.angular_diameter_distance_between_planes_in_units(i=1, j=1, unit_length='kpc') == 0.0
            assert tracer.angular_diameter_distance_between_planes_in_units(i=1, j=2, unit_length='kpc') == \
                   pytest.approx(638544, 1e-5)
            assert tracer.angular_diameter_distance_between_planes_in_units(i=1, j=3, unit_length='kpc') == \
                   pytest.approx(778472, 1e-5)

            assert tracer.arcsec_per_kpc_proper_of_plane(i=2) == pytest.approx(0.116500, 1e-5)
            assert tracer.kpc_per_arcsec_proper_of_plane(i=2) == pytest.approx(8.58368, 1e-5)

            assert tracer.angular_diameter_distance_of_plane_to_earth_in_units(i=2, unit_length='kpc') == \
                   pytest.approx(1770512, 1e-5)
            assert tracer.angular_diameter_distance_between_planes_in_units(i=2, j=0, unit_length='kpc') == \
                   pytest.approx(-4435831, 1e-5)
            assert tracer.angular_diameter_distance_between_planes_in_units(i=2, j=1, unit_length='kpc') == \
                   pytest.approx(-957816)
            assert tracer.angular_diameter_distance_between_planes_in_units(i=2, j=2, unit_length='kpc') == 0.0
            assert tracer.angular_diameter_distance_between_planes_in_units(i=2, j=3, unit_length='kpc') == \
                   pytest.approx(299564)

            assert tracer.arcsec_per_kpc_proper_of_plane(i=3) == pytest.approx(0.12674, 1e-5)
            assert tracer.kpc_per_arcsec_proper_of_plane(i=3) == pytest.approx(7.89009, 1e-5)

            assert tracer.angular_diameter_distance_of_plane_to_earth_in_units(i=3, unit_length='kpc') == \
                   pytest.approx(1627448, 1e-5)
            assert tracer.angular_diameter_distance_between_planes_in_units(i=3, j=0, unit_length='kpc') == \
                   pytest.approx(-5525155, 1e-5)
            assert tracer.angular_diameter_distance_between_planes_in_units(i=3, j=1, unit_length='kpc') == \
                   pytest.approx(-1556945, 1e-5)
            assert tracer.angular_diameter_distance_between_planes_in_units(i=3, j=2, unit_length='kpc') == \
                   pytest.approx(-399419, 1e-5)
            assert tracer.angular_diameter_distance_between_planes_in_units(i=3, j=3, unit_length='kpc') == 0.0

            assert tracer.critical_surface_density_between_planes_in_units(
                i=0, j=1, unit_length='kpc', unit_mass='solMass') == pytest.approx(4.85e9, 1e-2)
            assert tracer.critical_surface_density_between_planes_in_units(
                i=0, j=1, unit_length='arcsec', unit_mass='solMass') == pytest.approx(17593241668, 1e-2)

            assert tracer.scaling_factor_between_planes(i=0, j=1) == pytest.approx(0.9348, 1e-4)
            assert tracer.scaling_factor_between_planes(i=0, j=2) == pytest.approx(0.984, 1e-4)
            assert tracer.scaling_factor_between_planes(i=0, j=3) == pytest.approx(1.0, 1e-4)
            assert tracer.scaling_factor_between_planes(i=1, j=2) == pytest.approx(0.754, 1e-4)
            assert tracer.scaling_factor_between_planes(i=1, j=3) == pytest.approx(1.0, 1e-4)
            assert tracer.scaling_factor_between_planes(i=2, j=3) == pytest.approx(1.0, 1e-4)

    class TestPlaneSetup:

        def test__6_galaxies__tracer_planes_are_correct(self, grid_stack):
            g0 = g.Galaxy(redshift=2.0)
            g1 = g.Galaxy(redshift=2.0)
            g2 = g.Galaxy(redshift=0.1)
            g3 = g.Galaxy(redshift=3.0)
            g4 = g.Galaxy(redshift=1.0)
            g5 = g.Galaxy(redshift=3.0)

            tracer = ray_tracing.TracerMultiPlanes(galaxies=[g0, g1, g2, g3, g4, g5],
                                                   image_plane_grid_stack=grid_stack, cosmology=cosmo.Planck15)

            assert tracer.planes[0].galaxies == [g2]
            assert tracer.planes[1].galaxies == [g4]
            assert tracer.planes[2].galaxies == [g0, g1]
            assert tracer.planes[3].galaxies == [g3, g5]

    class TestPlaneGridStacks:

        def test__4_planes__data_grid_and_deflections_stacks_are_correct__sis_mass_profile(self, grid_stack):
            g0 = g.Galaxy(redshift=2.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(redshift=2.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g2 = g.Galaxy(redshift=0.1, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g3 = g.Galaxy(redshift=3.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g4 = g.Galaxy(redshift=1.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g5 = g.Galaxy(redshift=3.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))

            tracer = ray_tracing.TracerMultiPlanes(galaxies=[g0, g1, g2, g3, g4, g5],
                                                   image_plane_grid_stack=grid_stack, cosmology=cosmo.Planck15)

            # The scaling factors are as follows and were computed independently from the test.
            beta_01 = 0.9348
            beta_02 = 0.9839601
            # Beta_03 = 1.0
            beta_12 = 0.7539734
            # Beta_13 = 1.0
            # Beta_23 = 1.0

            val = np.sqrt(2) / 2.0

            assert tracer.planes[0].grid_stack.regular[0] == pytest.approx(np.array([1.0, 1.0]), 1e-4)
            assert tracer.planes[0].grid_stack.sub[0] == pytest.approx(np.array([1.0, 1.0]),
                                                                       1e-4)
            assert tracer.planes[0].grid_stack.sub[1] == pytest.approx(np.array([1.0, 0.0]),
                                                                       1e-4)
            assert tracer.planes[0].grid_stack.blurring[0] == pytest.approx(np.array([1.0, 0.0]),
                                                                            1e-4)
            assert tracer.planes[0].deflections_stack.regular[0] == pytest.approx(np.array([val, val]), 1e-4)
            assert tracer.planes[0].deflections_stack.sub[0] == pytest.approx(np.array([val, val]), 1e-4)
            assert tracer.planes[0].deflections_stack.sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-4)
            assert tracer.planes[0].deflections_stack.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-4)

            assert tracer.planes[1].grid_stack.regular[0] == pytest.approx(
                np.array([(1.0 - beta_01 * val), (1.0 - beta_01 * val)]), 1e-4)
            assert tracer.planes[1].grid_stack.sub[0] == pytest.approx(
                np.array([(1.0 - beta_01 * val), (1.0 - beta_01 * val)]), 1e-4)
            assert tracer.planes[1].grid_stack.sub[1] == pytest.approx(
                np.array([(1.0 - beta_01 * 1.0), 0.0]), 1e-4)
            assert tracer.planes[1].grid_stack.blurring[0] == pytest.approx(
                np.array([(1.0 - beta_01 * 1.0), 0.0]), 1e-4)

            defl11 = g0.deflections_from_grid(grid=np.array([[(1.0 - beta_01 * val), (1.0 - beta_01 * val)]]))
            defl12 = g0.deflections_from_grid(grid=np.array([[(1.0 - beta_01 * 1.0), 0.0]]))

            assert tracer.planes[1].deflections_stack.regular[0] == pytest.approx(defl11[0], 1e-4)
            assert tracer.planes[1].deflections_stack.sub[0] == pytest.approx(defl11[0], 1e-4)
            assert tracer.planes[1].deflections_stack.sub[1] == pytest.approx(defl12[0], 1e-4)
            assert tracer.planes[1].deflections_stack.blurring[0] == pytest.approx(defl12[0], 1e-4)

            assert tracer.planes[2].grid_stack.regular[0] == pytest.approx(
                np.array([(1.0 - beta_02 * val - beta_12 * defl11[0, 0]),
                          (1.0 - beta_02 * val - beta_12 * defl11[0, 1])]), 1e-4)
            assert tracer.planes[2].grid_stack.sub[0] == pytest.approx(
                np.array([(1.0 - beta_02 * val - beta_12 * defl11[0, 0]),
                          (1.0 - beta_02 * val - beta_12 * defl11[0, 1])]), 1e-4)
            assert tracer.planes[2].grid_stack.sub[1] == pytest.approx(
                np.array([(1.0 - beta_02 * 1.0 - beta_12 * defl12[0, 0]),
                          0.0]), 1e-4)
            assert tracer.planes[2].grid_stack.blurring[0] == pytest.approx(
                np.array([(1.0 - beta_02 * 1.0 - beta_12 * defl12[0, 0]),
                          0.0]), 1e-4)

            # 2 Galaxies in this plane, so multiply by 2.0

            defl21 = 2.0 * g0.deflections_from_grid(
                grid=np.array([[(1.0 - beta_02 * val - beta_12 * defl11[0, 0]),
                                (1.0 - beta_02 * val - beta_12 * defl11[0, 1])]]))
            defl22 = 2.0 * g0.deflections_from_grid(
                grid=np.array([[(1.0 - beta_02 * 1.0 - beta_12 * defl12[0, 0]),
                                0.0]]))

            assert tracer.planes[2].deflections_stack.regular[0] == pytest.approx(defl21[0], 1e-4)
            assert tracer.planes[2].deflections_stack.sub[0] == pytest.approx(defl21[0], 1e-4)
            assert tracer.planes[2].deflections_stack.sub[1] == pytest.approx(defl22[0], 1e-4)
            assert tracer.planes[2].deflections_stack.blurring[0] == pytest.approx(defl22[0], 1e-4)

            coord1 = (1.0 - tracer.planes[0].deflections_stack.regular[0, 0] - tracer.planes[1].deflections_stack.regular[
                0, 0] -
                      tracer.planes[2].deflections_stack.regular[0, 0])

            coord2 = (1.0 - tracer.planes[0].deflections_stack.regular[0, 1] - tracer.planes[1].deflections_stack.regular[
                0, 1] -
                      tracer.planes[2].deflections_stack.regular[0, 1])

            coord3 = (1.0 - tracer.planes[0].deflections_stack.sub[1, 0] -
                      tracer.planes[1].deflections_stack.sub[1, 0] -
                      tracer.planes[2].deflections_stack.sub[1, 0])

            assert tracer.planes[3].grid_stack.regular[0] == pytest.approx(np.array([coord1, coord2]),
                                                                           1e-4)
            assert tracer.planes[3].grid_stack.sub[0] == pytest.approx(
                np.array([coord1, coord2]), 1e-4)
            assert tracer.planes[3].grid_stack.sub[1] == pytest.approx(np.array([coord3, 0.0]),
                                                                       1e-4)
            assert tracer.planes[3].grid_stack.blurring[0] == pytest.approx(np.array([coord3, 0.0]),
                                                                            1e-4)

    class TestProfileImagePlaneImages:

        def test__x1_galaxy_light_no_mass_in_each_plane__image_of_each_plane_is_galaxy_image(self, grid_stack):
            g0 = g.Galaxy(redshift=0.1, light_profile=lp.EllipticalSersic(intensity=0.1))
            g1 = g.Galaxy(redshift=1.0, light_profile=lp.EllipticalSersic(intensity=0.2))
            g2 = g.Galaxy(redshift=2.0, light_profile=lp.EllipticalSersic(intensity=0.3))

            tracer = ray_tracing.TracerMultiPlanes(galaxies=[g0, g1, g2], image_plane_grid_stack=grid_stack,
                                                   cosmology=cosmo.Planck15)

            plane_0 = pl.Plane(galaxies=[g0], grid_stack=grid_stack, compute_deflections=True)
            plane_1 = pl.Plane(galaxies=[g1], grid_stack=grid_stack, compute_deflections=True)
            plane_2 = pl.Plane(galaxies=[g2], grid_stack=grid_stack, compute_deflections=False)

            image_plane_image = plane_0.profile_image_plane_image_2d + plane_1.profile_image_plane_image_2d + plane_2.profile_image_plane_image_2d

            assert image_plane_image.shape == (3, 4)
            assert (image_plane_image == tracer.profile_image_plane_image_2d).all()

        def test__galaxy_light_mass_sis__source_plane_image_includes_deflections(self, grid_stack):
            g0 = g.Galaxy(redshift=0.1, light_profile=lp.EllipticalSersic(intensity=0.1))
            g1 = g.Galaxy(redshift=1.0, light_profile=lp.EllipticalSersic(intensity=0.2))
            g2 = g.Galaxy(redshift=2.0, light_profile=lp.EllipticalSersic(intensity=0.3))

            tracer = ray_tracing.TracerMultiPlanes(galaxies=[g0, g1, g2], image_plane_grid_stack=grid_stack,
                                                   cosmology=cosmo.Planck15)

            plane_0 = tracer.planes[0]
            plane_1 = tracer.planes[1]
            plane_2 = tracer.planes[2]

            image_plane_image = plane_0.profile_image_plane_image_2d + plane_1.profile_image_plane_image_2d + plane_2.profile_image_plane_image_2d

            assert image_plane_image.shape == (3, 4)
            assert (image_plane_image == tracer.profile_image_plane_image_2d).all()

        def test__same_as_above_more_galaxies(self, grid_stack):
            g0 = g.Galaxy(redshift=0.1, light_profile=lp.EllipticalSersic(intensity=0.1))
            g1 = g.Galaxy(redshift=1.0, light_profile=lp.EllipticalSersic(intensity=0.2))
            g2 = g.Galaxy(redshift=2.0, light_profile=lp.EllipticalSersic(intensity=0.3))
            g3 = g.Galaxy(redshift=0.1, light_profile=lp.EllipticalSersic(intensity=0.4))
            g4 = g.Galaxy(redshift=1.0, light_profile=lp.EllipticalSersic(intensity=0.5))

            tracer = ray_tracing.TracerMultiPlanes(galaxies=[g0, g1, g2, g3, g4],
                                                   image_plane_grid_stack=grid_stack,
                                                   cosmology=cosmo.Planck15)

            plane_0 = pl.Plane(galaxies=[g0, g3], grid_stack=grid_stack, compute_deflections=True)
            plane_1 = pl.Plane(galaxies=[g1, g4], grid_stack=grid_stack, compute_deflections=True)
            plane_2 = pl.Plane(galaxies=[g2], grid_stack=grid_stack, compute_deflections=False)

            image_plane_image = plane_0.profile_image_plane_image_2d + plane_1.profile_image_plane_image_2d + plane_2.profile_image_plane_image_2d

            assert image_plane_image.shape == (3, 4)
            assert (image_plane_image == tracer.profile_image_plane_image_2d).all()

        def test__padded_2d_image_from_plane__mapped_correctly(self, padded_grid_stack):
            g0 = g.Galaxy(redshift=0.1, light_profile=lp.EllipticalSersic(intensity=0.1))
            g1 = g.Galaxy(redshift=1.0, light_profile=lp.EllipticalSersic(intensity=0.2))
            g2 = g.Galaxy(redshift=2.0, light_profile=lp.EllipticalSersic(intensity=0.3))

            tracer = ray_tracing.TracerMultiPlanes(galaxies=[g0, g1, g2], image_plane_grid_stack=padded_grid_stack,
                                                   cosmology=cosmo.Planck15)

            plane_0 = pl.Plane(galaxies=[g0], grid_stack=padded_grid_stack, compute_deflections=True)
            plane_1 = pl.Plane(galaxies=[g1], grid_stack=padded_grid_stack, compute_deflections=True)
            plane_2 = pl.Plane(galaxies=[g2], grid_stack=padded_grid_stack, compute_deflections=False)

            image_plane_image = plane_0.profile_image_plane_image_2d + plane_1.profile_image_plane_image_2d + plane_2.profile_image_plane_image_2d

            assert image_plane_image.shape == (1, 2)
            assert (image_plane_image == tracer.profile_image_plane_image_2d).all()

        def test__padded_2d_image_for_simulation__mapped_correctly_not_trimmed(self, padded_grid_stack):
            g0 = g.Galaxy(redshift=0.1, light_profile=lp.EllipticalSersic(intensity=0.1))
            g1 = g.Galaxy(redshift=1.0, light_profile=lp.EllipticalSersic(intensity=0.2))
            g2 = g.Galaxy(redshift=2.0, light_profile=lp.EllipticalSersic(intensity=0.3))

            tracer = ray_tracing.TracerMultiPlanes(galaxies=[g0, g1, g2], image_plane_grid_stack=padded_grid_stack,
                                                   cosmology=cosmo.Planck15)

            plane_0 = pl.Plane(galaxies=[g0], grid_stack=padded_grid_stack, compute_deflections=True)
            plane_1 = pl.Plane(galaxies=[g1], grid_stack=padded_grid_stack, compute_deflections=True)
            plane_2 = pl.Plane(galaxies=[g2], grid_stack=padded_grid_stack, compute_deflections=False)

            image_plane_image_for_simulation = (
                    plane_0.profile_image_plane_image_2d_for_simulation +
                    plane_1.profile_image_plane_image_2d_for_simulation +
                    plane_2.profile_image_plane_image_2d_for_simulation)

            assert image_plane_image_for_simulation.shape == (3, 4)
            assert (image_plane_image_for_simulation == tracer.profile_image_plane_image_2d_for_simulation).all()

    class TestProfileImagePlaneBlurringImages:

        def test__x1_galaxy_light_no_mass_in_each_plane__image_of_each_plane_is_galaxy_image(self, grid_stack):
            sersic = lp.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6, sersic_index=4.0)

            g0 = g.Galaxy(redshift=0.1, light_profile=sersic)
            g1 = g.Galaxy(redshift=1.0, light_profile=sersic)
            g2 = g.Galaxy(redshift=2.0, light_profile=sersic)

            tracer = ray_tracing.TracerMultiPlanes(galaxies=[g0, g1, g2], image_plane_grid_stack=grid_stack,
                                                   cosmology=cosmo.Planck15)

            plane_0 = pl.Plane(galaxies=[g0], grid_stack=grid_stack, compute_deflections=True)
            plane_1 = pl.Plane(galaxies=[g1], grid_stack=grid_stack, compute_deflections=True)
            plane_2 = pl.Plane(galaxies=[g2], grid_stack=grid_stack, compute_deflections=False)

            image_plane_blurring_image_1d = (
                    plane_0.profile_image_plane_blurring_image_1d +
                    plane_1.profile_image_plane_blurring_image_1d +
                    plane_2.profile_image_plane_blurring_image_1d)

            assert (image_plane_blurring_image_1d == tracer.profile_image_plane_blurring_image_1d).all()

        def test__galaxy_light_mass_sis__source_plane_image_includes_deflections(self, grid_stack):
            sersic = lp.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6, sersic_index=4.0)

            sis = mp.SphericalIsothermal(einstein_radius=1.0)

            g0 = g.Galaxy(redshift=0.1, light_profile=sersic, mass_profile=sis)
            g1 = g.Galaxy(redshift=1.0, light_profile=sersic, mass_profile=sis)
            g2 = g.Galaxy(redshift=2.0, light_profile=sersic, mass_profile=sis)

            tracer = ray_tracing.TracerMultiPlanes(galaxies=[g0, g1, g2], image_plane_grid_stack=grid_stack,
                                                   cosmology=cosmo.Planck15)

            plane_0 = tracer.planes[0]
            plane_1 = tracer.planes[1]
            plane_2 = tracer.planes[2]

            image_plane_blurring_image_1d = (plane_0.profile_image_plane_blurring_image_1d +
                                             plane_1.profile_image_plane_blurring_image_1d +
                                             plane_2.profile_image_plane_blurring_image_1d)

            assert (image_plane_blurring_image_1d == tracer.profile_image_plane_blurring_image_1d).all()

        def test__x1_galaxy_light_no_mass_in_each_plane__image_of_each_galaxy_is_galaxy_image(self, grid_stack):
            sersic = lp.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6, sersic_index=4.0)

            g0 = g.Galaxy(redshift=0.1, light_profile=sersic)
            g1 = g.Galaxy(redshift=1.0, light_profile=sersic)
            g2 = g.Galaxy(redshift=2.0, light_profile=sersic)

            tracer = ray_tracing.TracerMultiPlanes(galaxies=[g0, g1, g2], image_plane_grid_stack=grid_stack,
                                                   cosmology=cosmo.Planck15)

            plane_0 = pl.Plane(galaxies=[g0], grid_stack=grid_stack, compute_deflections=True)
            plane_1 = pl.Plane(galaxies=[g1], grid_stack=grid_stack, compute_deflections=True)
            plane_2 = pl.Plane(galaxies=[g2], grid_stack=grid_stack, compute_deflections=False)

            image_plane_blurring_image = (plane_0.profile_image_plane_blurring_image_1d +
                                          plane_1.profile_image_plane_blurring_image_1d +
                                          plane_2.profile_image_plane_blurring_image_1d)

            assert (image_plane_blurring_image == tracer.profile_image_plane_blurring_image_1d).all()

        def test__different_galaxies_no_mass_in_each_plane__image_of_each_galaxy_is_galaxy_image(self, grid_stack):
            sersic = lp.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6, sersic_index=4.0)

            g0 = g.Galaxy(redshift=0.1, light_profile=sersic)
            g1 = g.Galaxy(redshift=1.0, light_profile=sersic)
            g2 = g.Galaxy(redshift=2.0, light_profile=sersic)
            g3 = g.Galaxy(redshift=0.1, light_profile=sersic)
            g4 = g.Galaxy(redshift=1.0, light_profile=sersic)

            tracer = ray_tracing.TracerMultiPlanes(galaxies=[g0, g1, g2, g3, g4],
                                                   image_plane_grid_stack=grid_stack, cosmology=cosmo.Planck15)

            plane_0 = pl.Plane(galaxies=[g0, g3], grid_stack=grid_stack, compute_deflections=True)
            plane_1 = pl.Plane(galaxies=[g1, g4], grid_stack=grid_stack, compute_deflections=True)
            plane_2 = pl.Plane(galaxies=[g2], grid_stack=grid_stack, compute_deflections=False)

            image_plane_blurring_image_1d = (plane_0.profile_image_plane_blurring_image_1d +
                                             plane_1.profile_image_plane_blurring_image_1d +
                                             plane_2.profile_image_plane_blurring_image_1d)

            assert (image_plane_blurring_image_1d == tracer.profile_image_plane_blurring_image_1d).all()

    class TestImagePlanePixGrid:

        def test__galaxies_have_no_pixelization__no_pix_grid_added(self):
            mask = msk.Mask(np.array([[False, False, False],
                                      [False, False, False],
                                      [False, False, False]]), pixel_scale=1.0)

            grid_stack = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask, sub_grid_size=1,
                                                                                          psf_shape=(1, 1))

            tracer = ray_tracing.TracerMultiPlanes(galaxies=[g.Galaxy(redshift=2.0), g.Galaxy(redshift=1.0)],
                                                   image_plane_grid_stack=grid_stack)

            assert (tracer.planes[0].grid_stack.pix == np.array([[0.0, 0.0]])).all()
            assert (tracer.planes[1].grid_stack.pix == np.array([[0.0, 0.0]])).all()

        def test__setup_pixelization__galaxies_have_other_pixelization__returns_normal_grids(self):
            mask = msk.Mask(np.array([[False, False, False],
                                      [False, False, False],
                                      [False, False, False]]), pixel_scale=1.0)

            grid_stack = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask, sub_grid_size=1,
                                                                                          psf_shape=(1, 1))

            galaxy = g.Galaxy(redshift=2.0, pixelization=pixelizations.Rectangular(shape=(3, 3)),
                              regularization=regularization.Constant())

            tracer = ray_tracing.TracerMultiPlanes(galaxies=[galaxy, g.Galaxy(redshift=1.0)],
                                                   image_plane_grid_stack=grid_stack)

            assert (tracer.planes[0].grid_stack.pix == np.array([[0.0, 0.0]])).all()
            assert (tracer.planes[1].grid_stack.pix == np.array([[0.0, 0.0]])).all()

        def test__setup_pixelization__galaxy_has_pixelization__but_grid_is_padded_grid__returns_normal_grids(self):
            mask = msk.Mask(np.array([[False, False, False],
                                      [False, False, False],
                                      [False, True, False]]), pixel_scale=1.0)

            grid_stack = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                                 sub_grid_size=1,
                                                                                                 psf_shape=(1, 1))

            galaxy = g.Galaxy(redshift=2.0, pixelization=pixelizations.AdaptiveMagnification(shape=(3, 3)),
                              regularization=regularization.Constant())

            tracer = ray_tracing.TracerMultiPlanes(galaxies=[galaxy, g.Galaxy(redshift=1.0)],
                                                   image_plane_grid_stack=grid_stack)

            assert (tracer.planes[0].grid_stack.pix == np.array([[0.0, 0.0]])).all()
            assert (tracer.planes[1].grid_stack.pix == np.array([[0.0, 0.0]])).all()

        def test__setup_pixelization__galaxy_has_pixelization__returns_grids_with_pix_grid(self):
            mask = msk.Mask(np.array([[False, False, False],
                                      [False, False, False],
                                      [False, True, False]]), pixel_scale=1.0)

            grid_stack = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask, sub_grid_size=1,
                                                                                          psf_shape=(1, 1))

            galaxy = g.Galaxy(redshift=2.0, pixelization=pixelizations.AdaptiveMagnification(shape=(3, 3)),
                              regularization=regularization.Constant())

            tracer = ray_tracing.TracerMultiPlanes(galaxies=[galaxy, g.Galaxy(redshift=1.0)],
                                                   image_plane_grid_stack=grid_stack)

            assert (tracer.planes[0].grid_stack.regular == grid_stack.regular).all()
            assert (tracer.planes[0].grid_stack.sub == grid_stack.sub).all()
            assert (tracer.planes[0].grid_stack.blurring == grid_stack.blurring).all()
            assert (tracer.planes[0].grid_stack.pix == np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0],
                                                                 [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                                                 [-1.0, -1.0], [-1.0, 1.0]])).all()

            assert (tracer.planes[1].grid_stack.regular == grid_stack.regular).all()
            assert (tracer.planes[1].grid_stack.sub == grid_stack.sub).all()
            assert (tracer.planes[1].grid_stack.blurring == grid_stack.blurring).all()
            assert (tracer.planes[1].grid_stack.pix == np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0],
                                                                 [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                                                 [-1.0, -1.0], [-1.0, 1.0]])).all()


class TestMultiTracerFixedSlices(object):

    class TestCosmology:

        def test__4_planes_after_slicing(self, grid_stack):

            lens_g0 = g.Galaxy(redshift=0.5)
            source_g0 = g.Galaxy(redshift=2.0)
            los_g0 = g.Galaxy(redshift=1.0)

            tracer = ray_tracing.TracerMultiPlanesSliced(lens_galaxies=[lens_g0], line_of_sight_galaxies=[los_g0],
                                                         source_galaxies=[source_g0], planes_between_lenses=[1, 1],
                                                         image_plane_grid_stack=grid_stack,
                                                         cosmology=cosmo.Planck15)

            assert tracer.arcsec_per_kpc_proper_of_plane(i=0) == tracer.cosmology.arcsec_per_kpc_proper(z=0.25).value
            assert tracer.kpc_per_arcsec_proper_of_plane(i=0) == 1.0 / tracer.cosmology.arcsec_per_kpc_proper(
                z=0.25).value

            assert tracer.angular_diameter_distance_of_plane_to_earth_in_units(i=0, unit_length='kpc') == \
                   tracer.cosmology.angular_diameter_distance(0.25).to('kpc').value
            assert tracer.angular_diameter_distance_between_planes_in_units(i=0, j=0, unit_length='kpc') == \
                   tracer.cosmology.angular_diameter_distance_z1z2(0.25, 0.25).to('kpc').value
            assert tracer.angular_diameter_distance_between_planes_in_units(i=0,j=1, unit_length='kpc') == \
                   tracer.cosmology.angular_diameter_distance_z1z2(0.25, 0.5).to('kpc').value
            assert tracer.angular_diameter_distance_between_planes_in_units(i=0,j=2, unit_length='kpc') == \
                   tracer.cosmology.angular_diameter_distance_z1z2(0.25, 1.25).to('kpc').value
            assert tracer.angular_diameter_distance_between_planes_in_units(i=0, j=3, unit_length='kpc') == \
                   tracer.cosmology.angular_diameter_distance_z1z2(0.25, 2.0).to('kpc').value

            assert tracer.arcsec_per_kpc_proper_of_plane(i=1) == tracer.cosmology.arcsec_per_kpc_proper(z=0.5).value
            assert tracer.kpc_per_arcsec_proper_of_plane(i=1) == 1.0 / tracer.cosmology.arcsec_per_kpc_proper(z=0.5).value

            assert tracer.angular_diameter_distance_of_plane_to_earth_in_units(i=1, unit_length='kpc') == \
                   tracer.cosmology.angular_diameter_distance(0.5).to('kpc').value
            assert tracer.angular_diameter_distance_between_planes_in_units(i=1, j=0, unit_length='kpc') == \
                   tracer.cosmology.angular_diameter_distance_z1z2(0.5, 0.25).to('kpc').value
            assert tracer.angular_diameter_distance_between_planes_in_units(i=1,j=1, unit_length='kpc') == \
                   tracer.cosmology.angular_diameter_distance_z1z2(0.5, 0.5).to('kpc').value
            assert tracer.angular_diameter_distance_between_planes_in_units(i=1,j=2, unit_length='kpc') == \
                   tracer.cosmology.angular_diameter_distance_z1z2(0.5, 1.25).to('kpc').value
            assert tracer.angular_diameter_distance_between_planes_in_units(i=1,j=3, unit_length='kpc') == \
                   tracer.cosmology.angular_diameter_distance_z1z2(0.5, 2.0).to('kpc').value

            assert tracer.arcsec_per_kpc_proper_of_plane(i=2) == tracer.cosmology.arcsec_per_kpc_proper(z=1.25).value
            assert tracer.kpc_per_arcsec_proper_of_plane(i=2) == 1.0 / tracer.cosmology.arcsec_per_kpc_proper(z=1.25).value

            assert tracer.angular_diameter_distance_of_plane_to_earth_in_units(i=2, unit_length='kpc') == \
                   tracer.cosmology.angular_diameter_distance(1.25).to('kpc').value
            assert tracer.angular_diameter_distance_between_planes_in_units(i=2,j=0, unit_length='kpc') == \
                   tracer.cosmology.angular_diameter_distance_z1z2(1.25, 0.25).to('kpc').value
            assert tracer.angular_diameter_distance_between_planes_in_units(i=2,j=1, unit_length='kpc') == \
                   tracer.cosmology.angular_diameter_distance_z1z2(1.25, 0.5).to('kpc').value
            assert tracer.angular_diameter_distance_between_planes_in_units(i=2,j=2, unit_length='kpc') == \
                   tracer.cosmology.angular_diameter_distance_z1z2(1.25, 1.25).to('kpc').value
            assert tracer.angular_diameter_distance_between_planes_in_units(i=2,j=3, unit_length='kpc') == \
                   tracer.cosmology.angular_diameter_distance_z1z2(1.25, 2.0).to('kpc').value

            assert tracer.arcsec_per_kpc_proper_of_plane(i=3) == tracer.cosmology.arcsec_per_kpc_proper(z=2.0).value
            assert tracer.kpc_per_arcsec_proper_of_plane(i=3) == 1.0 / tracer.cosmology.arcsec_per_kpc_proper(
                z=2.0).value

            assert tracer.angular_diameter_distance_of_plane_to_earth_in_units(i=3, unit_length='kpc') == \
                   tracer.cosmology.angular_diameter_distance(2.0).to('kpc').value
            assert tracer.angular_diameter_distance_between_planes_in_units(i=3,j=0, unit_length='kpc') == \
                   tracer.cosmology.angular_diameter_distance_z1z2(2.0, 0.25).to('kpc').value
            assert tracer.angular_diameter_distance_between_planes_in_units(i=3,j=1, unit_length='kpc') == \
                   tracer.cosmology.angular_diameter_distance_z1z2(2.0, 0.5).to('kpc').value
            assert tracer.angular_diameter_distance_between_planes_in_units(i=3,j=2, unit_length='kpc') == \
                   tracer.cosmology.angular_diameter_distance_z1z2(2.0, 1.25).to('kpc').value
            assert tracer.angular_diameter_distance_between_planes_in_units(i=3,j=3, unit_length='kpc') == \
                   tracer.cosmology.angular_diameter_distance_z1z2(2.0, 2.0).to('kpc').value

    class TestPlaneSetup:

        def test__6_galaxies__tracer_planes_are_correct(self, grid_stack):
            lens_g0 = g.Galaxy(redshift=0.5)
            source_g0 = g.Galaxy(redshift=2.0)
            los_g0 = g.Galaxy(redshift=0.1)
            los_g1 = g.Galaxy(redshift=0.2)
            los_g2 = g.Galaxy(redshift=0.4)
            los_g3 = g.Galaxy(redshift=0.6)

            tracer = ray_tracing.TracerMultiPlanesSliced(lens_galaxies=[lens_g0],
                                                         line_of_sight_galaxies=[los_g0, los_g1, los_g2, los_g3],
                                                         source_galaxies=[source_g0], planes_between_lenses=[1, 1],
                                                         image_plane_grid_stack=grid_stack,
                                                         cosmology=cosmo.Planck15)

            # Plane redshifts are [0.25, 0.5, 1.25, 2.0]

            assert tracer.planes[0].galaxies == [los_g0, los_g1]
            assert tracer.planes[1].galaxies == [lens_g0, los_g2, los_g3]
            assert tracer.planes[2].galaxies == []
            assert tracer.planes[3].galaxies == [source_g0]

    class TestPlaneGridStacks:

        def test__4_planes__data_grid_and_deflections_stacks_are_correct__sis_mass_profile(self, grid_stack):
            lens_g0 = g.Galaxy(redshift=0.5, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            source_g0 = g.Galaxy(redshift=2.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            los_g0 = g.Galaxy(redshift=0.1, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            los_g1 = g.Galaxy(redshift=0.2, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            los_g2 = g.Galaxy(redshift=0.4, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            los_g3 = g.Galaxy(redshift=0.6, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))

            tracer = ray_tracing.TracerMultiPlanesSliced(lens_galaxies=[lens_g0],
                                                         line_of_sight_galaxies=[los_g0, los_g1, los_g2, los_g3],
                                                         source_galaxies=[source_g0], planes_between_lenses=[1, 1],
                                                         image_plane_grid_stack=grid_stack,
                                                         cosmology=cosmo.Planck15)

            # This test is essentially the same as the TracerMulti test, we just slightly change how many galaxies go 
            # in each plane and therefore change the factor in front of val for different planes.

            # The scaling factors are as follows and were computed indepedently from the test.
            beta_01 = 0.57874474423
            beta_02 = 0.91814281
            # Beta_03 = 1.0
            beta_12 = 0.8056827034
            # Beta_13 = 1.0
            # Beta_23 = 1.0

            val = np.sqrt(2) / 2.0

            assert tracer.planes[0].grid_stack.regular[0] == pytest.approx(np.array([1.0, 1.0]), 1e-4)
            assert tracer.planes[0].grid_stack.sub[0] == pytest.approx(np.array([1.0, 1.0]),
                                                                       1e-4)
            assert tracer.planes[0].grid_stack.sub[1] == pytest.approx(np.array([1.0, 0.0]),
                                                                       1e-4)
            assert tracer.planes[0].grid_stack.blurring[0] == pytest.approx(np.array([1.0, 0.0]),
                                                                            1e-4)
            assert tracer.planes[0].deflections_stack.regular[0] == pytest.approx(np.array([2.0 * val, 2.0 * val]), 1e-4)
            assert tracer.planes[0].deflections_stack.sub[0] == pytest.approx(np.array([2.0 * val, 2.0 * val]), 1e-4)
            assert tracer.planes[0].deflections_stack.sub[1] == pytest.approx(np.array([2.0, 0.0]), 1e-4)
            assert tracer.planes[0].deflections_stack.blurring[0] == pytest.approx(np.array([2.0, 0.0]), 1e-4)

            print(beta_01 * 2.0 * val)

            assert tracer.planes[1].grid_stack.regular[0] == pytest.approx(
                np.array([(1.0 - beta_01 * 2.0 * val), (1.0 - beta_01 * 2.0 * val)]), 1e-4)
            assert tracer.planes[1].grid_stack.sub[0] == pytest.approx(
                np.array([(1.0 - beta_01 * 2.0 * val), (1.0 - beta_01 * 2.0 * val)]), 1e-4)
            assert tracer.planes[1].grid_stack.sub[1] == pytest.approx(
                np.array([(1.0 - beta_01 * 2.0), 0.0]), 1e-4)
            assert tracer.planes[1].grid_stack.blurring[0] == pytest.approx(
                np.array([(1.0 - beta_01 * 2.0), 0.0]), 1e-4)

            #  Galaxies in this plane, so multiply by 3

            defl11 = 3.0 * lens_g0.deflections_from_grid(
                grid=np.array([[(1.0 - beta_01 * 2.0 * val), (1.0 - beta_01 * 2.0 * val)]]))
            defl12 = 3.0 * lens_g0.deflections_from_grid(grid=np.array([[(1.0 - beta_01 * 2.0 * 1.0), 0.0]]))

            assert tracer.planes[1].deflections_stack.regular[0] == pytest.approx(defl11[0], 1e-4)
            assert tracer.planes[1].deflections_stack.sub[0] == pytest.approx(defl11[0], 1e-4)
            assert tracer.planes[1].deflections_stack.sub[1] == pytest.approx(defl12[0], 1e-4)
            assert tracer.planes[1].deflections_stack.blurring[0] == pytest.approx(defl12[0], 1e-4)

            assert tracer.planes[2].grid_stack.regular[0] == pytest.approx(
                np.array([(1.0 - beta_02 * 2.0 * val - beta_12 * defl11[0, 0]),
                          (1.0 - beta_02 * 2.0 * val - beta_12 * defl11[0, 1])]), 1e-4)
            assert tracer.planes[2].grid_stack.sub[0] == pytest.approx(
                np.array([(1.0 - beta_02 * 2.0 * val - beta_12 * defl11[0, 0]),
                          (1.0 - beta_02 * 2.0 * val - beta_12 * defl11[0, 1])]), 1e-4)
            assert tracer.planes[2].grid_stack.sub[1] == pytest.approx(
                np.array([(1.0 - beta_02 * 2.0 - beta_12 * defl12[0, 0]), 0.0]), 1e-4)
            assert tracer.planes[2].grid_stack.blurring[0] == pytest.approx(
                np.array([(1.0 - beta_02 * 2.0 - beta_12 * defl12[0, 0]), 0.0]), 1e-4)

            # 0 Galaxies in this plane, so no defls

            defl21 = np.array([[0.0, 0.0]])
            defl22 = np.array([[0.0, 0.0]])

            assert tracer.planes[2].deflections_stack.regular[0] == pytest.approx(defl21[0], 1e-4)
            assert tracer.planes[2].deflections_stack.sub[0] == pytest.approx(defl21[0], 1e-4)
            assert tracer.planes[2].deflections_stack.sub[1] == pytest.approx(defl22[0], 1e-4)
            assert tracer.planes[2].deflections_stack.blurring[0] == pytest.approx(defl22[0], 1e-4)

            coord1 = (1.0 - tracer.planes[0].deflections_stack.regular[0, 0] -
                      tracer.planes[1].deflections_stack.regular[0, 0] -
                      tracer.planes[2].deflections_stack.regular[0, 0])

            coord2 = (1.0 - tracer.planes[0].deflections_stack.regular[0, 1] -
                      tracer.planes[1].deflections_stack.regular[0, 1] -
                      tracer.planes[2].deflections_stack.regular[0, 1])

            coord3 = (1.0 - tracer.planes[0].deflections_stack.sub[1, 0] -
                      tracer.planes[1].deflections_stack.sub[1, 0] -
                      tracer.planes[2].deflections_stack.sub[1, 0])

            assert tracer.planes[3].grid_stack.regular[0] == pytest.approx(np.array([coord1, coord2]), 1e-4)
            assert tracer.planes[3].grid_stack.sub[0] == pytest.approx(np.array([coord1, coord2]), 1e-4)
            assert tracer.planes[3].grid_stack.sub[1] == pytest.approx(np.array([coord3, 0.0]), 1e-4)
            assert tracer.planes[3].grid_stack.blurring[0] == pytest.approx(np.array([coord3, 0.0]), 1e-4)


class TestTracerImageAndSourcePositions(object):
    class TestSetup:

        def test__x2_positions__no_galaxy__image_and_source_planes_setup__same_positions(self, galaxy_non):
            tracer = ray_tracing.TracerImageSourcePlanesPositions(lens_galaxies=[galaxy_non],
                                                                  image_plane_positions=[
                                                                      np.array([[1.0, 1.0], [-1.0, -1.0]])])

            assert tracer.image_plane.positions[0] == pytest.approx(np.array([[1.0, 1.0], [-1.0, -1.0]]), 1e-3)
            assert tracer.image_plane.deflections[0] == pytest.approx(np.array([[0.0, 0.0], [0.0, 0.0]]), 1e-3)
            assert tracer.source_plane.positions[0] == pytest.approx(np.array([[1.0, 1.0], [-1.0, -1.0]]), 1e-3)

        def test__x2_positions__sis_lens__positions_with_source_plane_deflected(self, galaxy_mass):
            tracer = ray_tracing.TracerImageSourcePlanesPositions(lens_galaxies=[galaxy_mass],
                                                                  image_plane_positions=[
                                                                      np.array([[1.0, 1.0], [-1.0, -1.0]])])

            assert tracer.image_plane.positions[0] == pytest.approx(np.array([[1.0, 1.0], [-1.0, -1.0]]), 1e-3)
            assert tracer.image_plane.deflections[0] == pytest.approx(np.array([[0.707, 0.707], [-0.707, -0.707]]),
                                                                      1e-3)
            assert tracer.source_plane.positions[0] == pytest.approx(np.array([[1.0 - 0.707, 1.0 - 0.707],
                                                                               [-1.0 + 0.707, -1.0 + 0.707]]), 1e-3)

        def test__same_as_above_but_2_sis_lenses__deflections_double(self, galaxy_mass):
            tracer = ray_tracing.TracerImageSourcePlanesPositions(lens_galaxies=[galaxy_mass, galaxy_mass],
                                                                  image_plane_positions=[
                                                                      np.array([[1.0, 1.0], [-1.0, -1.0]])])

            assert tracer.image_plane.positions[0] == pytest.approx(np.array([[1.0, 1.0], [-1.0, -1.0]]), 1e-3)
            assert tracer.image_plane.deflections[0] == pytest.approx(np.array([[1.414, 1.414], [-1.414, -1.414]]),
                                                                      1e-3)
            assert tracer.source_plane.positions[0] == pytest.approx(np.array([[1.0 - 1.414, 1.0 - 1.414],
                                                                               [-1.0 + 1.414, -1.0 + 1.414]]), 1e-3)

        def test__multiple_sets_of_positions_in_different_arrays(self, galaxy_mass):
            tracer = ray_tracing.TracerImageSourcePlanesPositions(lens_galaxies=[galaxy_mass],
                                                                  image_plane_positions=[
                                                                      np.array([[1.0, 1.0], [-1.0, -1.0]]),
                                                                      np.array([[0.5, 0.5]])])

            assert tracer.image_plane.positions[0] == pytest.approx(np.array([[1.0, 1.0], [-1.0, -1.0]]), 1e-3)
            assert tracer.image_plane.deflections[0] == pytest.approx(np.array([[0.707, 0.707], [-0.707, -0.707]]),
                                                                      1e-3)
            assert tracer.source_plane.positions[0] == pytest.approx(np.array([[1.0 - 0.707, 1.0 - 0.707],
                                                                               [-1.0 + 0.707, -1.0 + 0.707]]), 1e-3)

            assert tracer.image_plane.positions[1] == pytest.approx(np.array([[0.5, 0.5]]), 1e-3)
            assert tracer.image_plane.deflections[1] == pytest.approx(np.array([[0.707, 0.707]]), 1e-3)
            assert tracer.source_plane.positions[1] == pytest.approx(np.array([[0.5 - 0.707, 0.5 - 0.707]]), 1e-3)


class TestTracerMultiPositions(object):
    class TestRayTracingPlanes:

        def test__4_planes__coordinate_grid_stack_and_deflections_are_correct__sis_mass_profile(self):
            import math

            g0 = g.Galaxy(redshift=2.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(redshift=2.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g2 = g.Galaxy(redshift=0.1, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g3 = g.Galaxy(redshift=3.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g4 = g.Galaxy(redshift=1.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g5 = g.Galaxy(redshift=3.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))

            tracer = ray_tracing.TracerMultiPlanesPositions(galaxies=[g0, g1, g2, g3, g4, g5],
                                                            image_plane_positions=[np.array([[1.0, 1.0]])],
                                                            cosmology=cosmo.Planck15)

            # From unit test below:
            # Beta_01 = 0.9348
            beta_02 = 0.9839601
            # Beta_03 = 1.0
            beta_12 = 0.7539734
            # Beta_13 = 1.0
            # Beta_23 = 1.0

            val = math.sqrt(2) / 2.0

            assert tracer.planes[0].positions[0] == pytest.approx(np.array([[1.0, 1.0]]), 1e-4)
            assert tracer.planes[0].deflections[0] == pytest.approx(np.array([[val, val]]), 1e-4)

            assert tracer.planes[1].positions[0] == pytest.approx(
                np.array([[(1.0 - 0.9348 * val), (1.0 - 0.9348 * val)]]), 1e-4)

            defl11 = g0.deflections_from_grid(grid=np.array([[(1.0 - 0.9348 * val), (1.0 - 0.9348 * val)]]))

            assert tracer.planes[1].deflections[0] == pytest.approx(defl11[[0]], 1e-4)

            assert tracer.planes[2].positions[0] == pytest.approx(
                np.array([[(1.0 - beta_02 * val - beta_12 * defl11[0, 0]),
                           (1.0 - beta_02 * val - beta_12 * defl11[0, 1])]]), 1e-4)

            # 2 Galaxies in this plane, so multiply by 2.0

            defl21 = 2.0 * g0.deflections_from_grid(grid=np.array([[(1.0 - beta_02 * val - beta_12 * defl11[0, 0]),
                                                                    (1.0 - beta_02 * val - beta_12 * defl11[
                                                                        0, 1])]]))

            assert tracer.planes[2].deflections[0] == pytest.approx(defl21[[0]], 1e-4)

            coord1 = (1.0 - tracer.planes[0].deflections[0][0, 0] - tracer.planes[1].deflections[0][0, 0] -
                      tracer.planes[2].deflections[0][0, 0])

            coord2 = (1.0 - tracer.planes[0].deflections[0][0, 1] - tracer.planes[1].deflections[0][0, 1] -
                      tracer.planes[2].deflections[0][0, 1])

            assert tracer.planes[3].positions[0] == pytest.approx(np.array([[coord1, coord2]]), 1e-4)

        def test__same_as_above_but_multiple_sets_of_positions(self):
            import math

            g0 = g.Galaxy(redshift=2.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(redshift=2.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g2 = g.Galaxy(redshift=0.1, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g3 = g.Galaxy(redshift=3.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g4 = g.Galaxy(redshift=1.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g5 = g.Galaxy(redshift=3.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))

            tracer = ray_tracing.TracerMultiPlanesPositions(galaxies=[g0, g1, g2, g3, g4, g5],
                                                            image_plane_positions=[np.array([[1.0, 1.0]]),
                                                                                   np.array([[1.0, 1.0]])],
                                                            cosmology=cosmo.Planck15)

            # From unit test below:
            # Beta_01 = 0.9348
            beta_02 = 0.9839601
            # Beta_03 = 1.0
            beta_12 = 0.7539734
            # Beta_13 = 1.0
            # Beta_23 = 1.0

            val = math.sqrt(2) / 2.0

            assert tracer.planes[0].positions[0] == pytest.approx(np.array([[1.0, 1.0]]), 1e-4)
            assert tracer.planes[0].deflections[0] == pytest.approx(np.array([[val, val]]), 1e-4)

            assert tracer.planes[1].positions[0] == pytest.approx(
                np.array([[(1.0 - 0.9348 * val), (1.0 - 0.9348 * val)]]), 1e-4)

            defl11 = g0.deflections_from_grid(grid=np.array([[(1.0 - 0.9348 * val), (1.0 - 0.9348 * val)]]))

            assert tracer.planes[1].deflections[0] == pytest.approx(defl11[[0]], 1e-4)

            assert tracer.planes[2].positions[0] == pytest.approx(
                np.array([[(1.0 - beta_02 * val - beta_12 * defl11[0, 0]),
                           (1.0 - beta_02 * val - beta_12 * defl11[0, 1])]]), 1e-4)

            # 2 Galaxies in this plane, so multiply by 2.0

            defl21 = 2.0 * g0.deflections_from_grid(grid=np.array([[(1.0 - beta_02 * val - beta_12 * defl11[0, 0]),
                                                                    (1.0 - beta_02 * val - beta_12 * defl11[
                                                                        0, 1])]]))

            assert tracer.planes[2].deflections[0] == pytest.approx(defl21[[0]], 1e-4)

            coord1 = (1.0 - tracer.planes[0].deflections[0][0, 0] - tracer.planes[1].deflections[0][0, 0] -
                      tracer.planes[2].deflections[0][0, 0])

            coord2 = (1.0 - tracer.planes[0].deflections[0][0, 1] - tracer.planes[1].deflections[0][0, 1] -
                      tracer.planes[2].deflections[0][0, 1])

            assert tracer.planes[3].positions[0] == pytest.approx(np.array([[coord1, coord2]]), 1e-4)

            assert tracer.planes[0].positions[1] == pytest.approx(np.array([[1.0, 1.0]]), 1e-4)
            assert tracer.planes[0].deflections[1] == pytest.approx(np.array([[val, val]]), 1e-4)

            assert tracer.planes[1].positions[1] == pytest.approx(
                np.array([[(1.0 - 0.9348 * val), (1.0 - 0.9348 * val)]]), 1e-4)

            defl11 = g0.deflections_from_grid(grid=np.array([[(1.0 - 0.9348 * val), (1.0 - 0.9348 * val)]]))

            assert tracer.planes[1].deflections[1] == pytest.approx(defl11[[0]], 1e-4)
            assert tracer.planes[2].positions[1] == pytest.approx(
                np.array([[(1.0 - beta_02 * val - beta_12 * defl11[0, 0]),
                           (1.0 - beta_02 * val - beta_12 * defl11[0, 1])]]), 1e-4)

            # 2 Galaxies in this plane, so multiply by 2.0

            defl21 = 2.0 * g0.deflections_from_grid(grid=np.array([[(1.0 - beta_02 * val - beta_12 * defl11[0, 0]),
                                                                    (1.0 - beta_02 * val - beta_12 * defl11[
                                                                        0, 1])]]))

            assert tracer.planes[2].deflections[1] == pytest.approx(defl21[[0]], 1e-4)

            coord1 = (1.0 - tracer.planes[0].deflections[1][0, 0] - tracer.planes[1].deflections[1][0, 0] -
                      tracer.planes[2].deflections[1][0, 0])

            coord2 = (1.0 - tracer.planes[0].deflections[1][0, 1] - tracer.planes[1].deflections[1][0, 1] -
                      tracer.planes[2].deflections[1][0, 1])

            assert tracer.planes[3].positions[1] == pytest.approx(np.array([[coord1, coord2]]), 1e-4)
