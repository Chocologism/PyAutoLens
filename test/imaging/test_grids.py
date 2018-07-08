from __future__ import division, print_function
import pytest
import numpy as np
from src.imaging import grids
from src.imaging import mask as msk
from src.imaging import image as img

from src.analysis import galaxy
from src.profiles import light_profiles, mass_profiles
import os

test_data_dir = "{}/../../weighted_data/test_data/".format(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(name='galaxy_no_profiles', scope='function')
def make_galaxy_no_profiles():
    return galaxy.Galaxy()


@pytest.fixture(name='galaxy_mass_sis', scope='function')
def make_galaxy_mass_sis():
    sis = mass_profiles.SphericalIsothermal(einstein_radius=1.0)
    return galaxy.Galaxy(mass_profile=sis)


@pytest.fixture(name='galaxy_light_sersic', scope='function')
def make_galaxy_light_sersic():
    sersic = light_profiles.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6,
                                             sersic_index=4.0)
    return galaxy.Galaxy(light_profile=sersic)


@pytest.fixture(name='lens_sis_x3')
def make_lens_sis_x3():
    mass_profile = mass_profiles.SphericalIsothermal(einstein_radius=1.0)
    return galaxy.Galaxy(mass_profile_1=mass_profile, mass_profile_2=mass_profile,
                         mass_profile_3=mass_profile)


class TestCoordsCollection(object):

    class TestConstructor(object):

        def test__simple_grid_input__all_grids_used__sets_up_attributes(self):

            image_grid = grids.CoordinateGrid(np.array([[1.0, 1.0],
                                                        [2.0, 2.0],
                                                        [3.0, 3.0]]))

            sub_grid = grids.SubCoordinateGrid(np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0],
                                                         [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]]), sub_grid_size=2)

            blurring_grid = grids.CoordinateGrid(np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0],
                                                           [1.0, 1.0]]))

            grid_collection = grids.CoordsCollection(image_grid, sub_grid, blurring_grid)

            assert (grid_collection.image[0] == np.array([1.0, 1.0])).all()
            assert (grid_collection.image[1] == np.array([2.0, 2.0])).all()
            assert (grid_collection.image[2] == np.array([3.0, 3.0])).all()

            assert (grid_collection.sub[0] == np.array([1.0, 1.0])).all()
            assert (grid_collection.sub[1] == np.array([1.0, 1.0])).all()
            assert (grid_collection.sub[2] == np.array([1.0, 1.0])).all()
            assert (grid_collection.sub[3] == np.array([1.0, 1.0])).all()
            assert (grid_collection.sub[4] == np.array([2.0, 2.0])).all()
            assert (grid_collection.sub[5] == np.array([2.0, 2.0])).all()
            assert (grid_collection.sub[6] == np.array([2.0, 2.0])).all()
            assert (grid_collection.sub[7] == np.array([2.0, 2.0])).all()

            assert (grid_collection.blurring[0] == np.array([1.0, 1.0])).all()
            assert (grid_collection.blurring[0] == np.array([1.0, 1.0])).all()
            assert (grid_collection.blurring[0] == np.array([1.0, 1.0])).all()
            assert (grid_collection.blurring[0] == np.array([1.0, 1.0])).all()

    class TestFromMask(object):

        def test__all_grids_from_masks__correct_grids_setup(self):

            mask = np.array([[True, True, True],
                             [True, False, True],
                             [True, True, True]])

            mask = msk.Mask(array=mask, pixel_scale=3.0)

            image_grid = mask.coordinate_grid
            sub_grid = mask.sub_coordinate_grid_with_size(size=2)
            blurring_grid = mask.blurring_coordinate_grid(psf_size=(3, 3))

            grid_collection = grids.CoordsCollection.from_mask(mask, sub_grid_size=2, blurring_shape=(3, 3))

            assert (grid_collection.image == image_grid).all()
            assert (grid_collection.sub == sub_grid).all()
            assert (grid_collection.blurring == blurring_grid).all()

    class TestDeflectionAnglesViaGalaxy(object):

        def test_all_coordinates(self, galaxy_mass_sis):

            image_grid = np.array([[1.0, 1.0]])
            sub_grid = np.array([[1.0, 1.0], [1.0, 0.0]])
            blurring_grid = np.array([[1.0, 0.0]])

            image_grid = grids.CoordinateGrid(image_grid)
            sub_grid = grids.SubCoordinateGrid(sub_grid, sub_grid_size=2)
            blurring_grid = grids.CoordinateGrid(blurring_grid)

            ray_trace_grid = grids.CoordsCollection(image=image_grid, sub=sub_grid, blurring=blurring_grid)

            deflections = ray_trace_grid.deflection_grids_for_galaxies([galaxy_mass_sis])

            assert deflections.image[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert deflections.sub[0] == pytest.approx(np.array([0.707, 0.707]), 1e-3)
            assert deflections.sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert deflections.sub.sub_grid_size == 2
            assert deflections.blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

        def test_three_identical_lenses__deflection_angles_triple(self, galaxy_mass_sis):
            image_grid = np.array([[1.0, 1.0]])
            sub_grid = np.array([[1.0, 1.0], [1.0, 0.0]])
            blurring_grid = np.array([[1.0, 0.0]])

            image_grid = grids.CoordinateGrid(image_grid)
            sub_grid = grids.SubCoordinateGrid(sub_grid, sub_grid_size=2)
            blurring_grid = grids.CoordinateGrid(blurring_grid)

            ray_trace_grid = grids.CoordsCollection(image=image_grid, sub=sub_grid, blurring=blurring_grid)

            deflections = ray_trace_grid.deflection_grids_for_galaxies(
                [galaxy_mass_sis, galaxy_mass_sis, galaxy_mass_sis])

            assert deflections.image == pytest.approx(np.array([[3.0 * 0.707, 3.0 * 0.707]]), 1e-3)
            assert deflections.sub[0] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]), 1e-3)
            assert deflections.sub[1] == pytest.approx(np.array([3.0, 0.0]), 1e-3)
            assert deflections.sub.sub_grid_size == 2
            assert deflections.blurring[0] == pytest.approx(np.array([3.0, 0.0]), 1e-3)

        def test_one_lens_with_three_identical_mass_profiles__deflection_angles_triple(self, lens_sis_x3):

            image_grid = np.array([[1.0, 1.0]])
            sub_grid = np.array([[1.0, 1.0], [1.0, 0.0]])
            blurring_grid = np.array([[1.0, 0.0]])

            image_grid = grids.CoordinateGrid(image_grid)
            sub_grid = grids.SubCoordinateGrid(sub_grid, sub_grid_size=2)
            blurring_grid = grids.CoordinateGrid(blurring_grid)

            ray_trace_grid = grids.CoordsCollection(image=image_grid, sub=sub_grid, blurring=blurring_grid)

            deflections = ray_trace_grid.deflection_grids_for_galaxies([lens_sis_x3])

            assert deflections.image == pytest.approx(np.array([[3.0 * 0.707, 3.0 * 0.707]]), 1e-3)
            assert deflections.sub[0] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]), 1e-3)
            assert deflections.sub[1] == pytest.approx(np.array([3.0, 0.0]), 1e-3)
            assert deflections.sub.sub_grid_size == 2
            assert deflections.blurring[0] == pytest.approx(np.array([3.0, 0.0]), 1e-3)

        # TODO: Removed to avoid testing quad galaxies. Reintroduce
        # def test__complex_mass_model(self):
        #     image_grid = np.array([[1.0, 1.0]])
        #     sub_grid = np.array([[1.0, 1.0], [1.0, 0.0]])
        #     blurring_grid = np.array([[1.0, 0.0]])
        #
        #     image_grid = grids.CoordinateGrid(image_grid)
        #     sub_grid = grids.SubCoordinateGrid(sub_grid, sub_grid_size=4, sub_to_image=np.array([0, 0]), image_pixels=1)
        #     blurring_grid = grids.CoordinateGrid(blurring_grid)
        #
        #     power_law = mass_profiles.EllipticalPowerLaw(centre=(1.0, 4.0), axis_ratio=0.7, phi=30.0,
        #                                                  einstein_radius=1.0, slope=2.2)
        #
        #     nfw = mass_profiles.SphericalNFW(kappa_s=0.1, scale_radius=5.0)
        #
        #     lens_galaxy = galaxy.Galaxy(redshift=0.1, mass_profile_1=power_law, mass_profile_2=nfw)
        #
        #     ray_trace_grid = grids.CoordsCollection(image=image_grid, sub=sub_grid, blurring=blurring_grid)
        #
        #     deflections = ray_trace_grid.deflection_grids_for_galaxies([lens_galaxy])
        #
        #     defls = power_law.deflections_at_coordinates(image_grid[0]) + \
        #             nfw.deflections_at_coordinates(image_grid[0])
        #
        #     sub_defls_0 = power_law.deflections_at_coordinates(sub_grid[0, 0]) + \
        #                   nfw.deflections_at_coordinates(sub_grid[0, 0])
        #
        #     sub_defls_1 = power_law.deflections_at_coordinates(sub_grid[0, 1]) + \
        #                   nfw.deflections_at_coordinates(sub_grid[0, 1])
        #
        #     blurring_defls = power_law.deflections_at_coordinates(blurring_grid[0]) + \
        #                      nfw.deflections_at_coordinates(blurring_grid[0])
        #
        #     assert deflections.image[0] == pytest.approx(defls, 1e-3)
        #     assert deflections.sub[0] == pytest.approx(sub_defls_0, 1e-3)
        #     assert deflections.sub[1] == pytest.approx(sub_defls_1, 1e-3)
        #     assert deflections.sub.sub_grid_size == 4
        #     assert deflections.blurring[0] == pytest.approx(blurring_defls, 1e-3)

    class TestSetupTracedGrids(object):

        def test_all_coordinates(self, galaxy_mass_sis):

            image_grid = np.array([[1.0, 1.0]])
            sub_grid = np.array([[1.0, 1.0], [1.0, 0.0]])
            blurring_grid = np.array([[1.0, 0.0]])

            image_grid = grids.CoordinateGrid(image_grid)
            sub_grid = grids.SubCoordinateGrid(sub_grid, sub_grid_size=2)
            blurring_grid = grids.CoordinateGrid(blurring_grid)

            ray_trace_grid = grids.CoordsCollection(image=image_grid, sub=sub_grid, blurring=blurring_grid)

            deflections = ray_trace_grid.deflection_grids_for_galaxies([galaxy_mass_sis])

            traced = ray_trace_grid.traced_grids_for_deflections(deflections)

            assert traced.image[0] == pytest.approx(np.array([1.0 - 0.707, 1.0 - 0.707]), 1e-3)
            assert traced.sub[0] == pytest.approx(np.array([1.0 - 0.707, 1.0 - 0.707]), 1e-3)
            assert traced.sub[1] == pytest.approx(np.array([1.0 - 1.0, 0.0 - 0.0]), 1e-3)
            assert traced.sub.sub_grid_size == 2
            assert traced.blurring[0] == pytest.approx(np.array([1.0 - 1.0, 0.0 - 0.0]), 1e-3)


class TestCoordinateGrid(object):

    class TestConstructor:

        def test__simple_grid_input__sets_up_grid_correctly_in_attributes(self):
            regular_grid_coords = np.array([[1.0, 1.0],
                                            [2.0, 2.0],
                                            [3.0, 3.0]])

            grid_regular = grids.CoordinateGrid(regular_grid_coords)

            assert (grid_regular[0] == np.array([1.0, 1.0])).all()
            assert (grid_regular[1] == np.array([2.0, 2.0])).all()
            assert (grid_regular[2] == np.array([3.0, 3.0])).all()

        def test__type(self):
            regular_grid_coords = np.array([[1.0, 1.0],
                                            [2.0, 2.0],
                                            [3.0, 3.0]])

            grid_regular = grids.CoordinateGrid(regular_grid_coords)
            assert isinstance(grid_regular, grids.CoordinateGrid)

    class TestIntensityViaGrid:

        def test__no_galaxies__intensities_returned_as_0s(self, galaxy_no_profiles):
            regular_grid_coords = np.array([[1.0, 1.0],
                                            [2.0, 2.0],
                                            [3.0, 3.0]])

            grid_regular = grids.CoordinateGrid(regular_grid_coords)

            intensities = grid_regular.intensities_via_grid(galaxies=[galaxy_no_profiles])

            assert (intensities[0] == np.array([0.0, 0.0])).all()
            assert (intensities[1] == np.array([0.0, 0.0])).all()
            assert (intensities[2] == np.array([0.0, 0.0])).all()

        def test__galaxy_sersic_light__intensities_returned_as_correct_values(self, galaxy_light_sersic):
            regular_grid_coords = np.array([[1.0, 1.0],
                                            [1.0, 0.0],
                                            [-1.0, 0.0]])

            intensity_0 = galaxy_light_sersic.intensity_at_coordinates(regular_grid_coords[0])
            intensity_1 = galaxy_light_sersic.intensity_at_coordinates(regular_grid_coords[1])
            intensity_2 = galaxy_light_sersic.intensity_at_coordinates(regular_grid_coords[2])

            grid_regular = grids.CoordinateGrid(regular_grid_coords)

            intensities = grid_regular.intensities_via_grid(galaxies=[galaxy_light_sersic])

            assert intensities[0] == intensity_0
            assert intensities[1] == intensity_1
            assert intensities[2] == intensity_2

        def test__galaxy_sis_mass_x3__intensities_tripled_from_above(self, galaxy_light_sersic):
            regular_grid_coords = np.array([[1.0, 1.0],
                                            [1.0, 0.0],
                                            [-1.0, 0.0]])

            intensity_0 = galaxy_light_sersic.intensity_at_coordinates(regular_grid_coords[0])
            intensity_1 = galaxy_light_sersic.intensity_at_coordinates(regular_grid_coords[1])
            intensity_2 = galaxy_light_sersic.intensity_at_coordinates(regular_grid_coords[2])

            grid_regular = grids.CoordinateGrid(regular_grid_coords)

            intensities = grid_regular.intensities_via_grid(
                galaxies=[galaxy_light_sersic, galaxy_light_sersic, galaxy_light_sersic])

            assert intensities[0] == 3.0 * intensity_0
            assert intensities[1] == 3.0 * intensity_1
            assert intensities[2] == 3.0 * intensity_2

    class TestDeflectionsOnGrid:

        def test__no_galaxies__deflections_returned_as_0s(self, galaxy_no_profiles):
            regular_grid_coords = np.array([[1.0, 1.0],
                                            [2.0, 2.0],
                                            [3.0, 3.0]])

            grid_regular = grids.CoordinateGrid(regular_grid_coords)

            deflections = grid_regular.deflections_on_grid(galaxies=[galaxy_no_profiles])

            assert (deflections[0] == np.array([0.0, 0.0])).all()
            assert (deflections[1] == np.array([0.0, 0.0])).all()
            assert (deflections[2] == np.array([0.0, 0.0])).all()

        def test__galaxy_sis_mass__deflections_returned_as_correct_values(self, galaxy_mass_sis):
            regular_grid_coords = np.array([[1.0, 1.0],
                                            [1.0, 0.0],
                                            [-1.0, 0.0]])

            grid_regular = grids.CoordinateGrid(regular_grid_coords)

            deflections = grid_regular.deflections_on_grid(galaxies=[galaxy_mass_sis])

            assert deflections[0] == pytest.approx(np.array([0.707, 0.707]), 1e-2)
            assert deflections[1] == pytest.approx(np.array([1.0, 0.0]), 1e-2)
            assert deflections[2] == pytest.approx(np.array([-1.0, 0.0]), 1e-2)

        def test__galaxy_sis_mass_x3__deflections_tripled_from_above(self, galaxy_mass_sis):
            regular_grid_coords = np.array([[1.0, 1.0],
                                            [1.0, 0.0],
                                            [-1.0, 0.0]])

            grid_regular = grids.CoordinateGrid(regular_grid_coords)

            deflections = grid_regular.deflections_on_grid(galaxies=[galaxy_mass_sis, galaxy_mass_sis, galaxy_mass_sis])

            assert deflections[0] == pytest.approx(3.0 * np.array([0.707, 0.707]), 1e-2)
            assert deflections[1] == pytest.approx(3.0 * np.array([1.0, 0.0]), 1e-2)
            assert deflections[2] == pytest.approx(3.0 * np.array([-1.0, 0.0]), 1e-2)

    class TestDeflectionsForGalaxies:

        def test__simple_sis_model__deflection_angles(self, galaxy_mass_sis):

            regular_grid_coords = np.array([[1.0, 1.0], [-1.0, -1.0]])

            grid_image = grids.CoordinateGrid(regular_grid_coords)

            grid_deflections = grid_image.deflection_grid_for_galaxies(galaxies=[galaxy_mass_sis])

            assert grid_deflections[0] == pytest.approx(np.array([0.707, 0.707]), 1e-2)
            assert grid_deflections[1] == pytest.approx(np.array([-0.707, -0.707]), 1e-2)

        def test_three_identical_lenses__deflection_angles_triple(self, galaxy_mass_sis):
            grid_image = np.array([[1.0, 1.0]])

            grid_image = grids.CoordinateGrid(grid_image)

            grid_deflections = grid_image.deflection_grid_for_galaxies(
                galaxies=[galaxy_mass_sis, galaxy_mass_sis, galaxy_mass_sis])

            assert grid_deflections == pytest.approx(np.array([[3.0 * 0.707, 3.0 * 0.707]]), 1e-3)

        def test_one_lens_with_three_identical_mass_profiles__deflection_angles_triple(self, lens_sis_x3):
            grid_image = np.array([[1.0, 1.0]])

            grid_image = grids.CoordinateGrid(grid_image)

            grid_deflections = grid_image.deflection_grid_for_galaxies(galaxies=[lens_sis_x3])

            assert grid_deflections == pytest.approx(np.array([[3.0 * 0.707, 3.0 * 0.707]]), 1e-3)

    class TestSetupTracedGrid:

        def test__simple_sis_model__deflection_angles(self, galaxy_mass_sis):
            regular_grid_coords = np.array([[1.0, 1.0],
                                            [-1.0, -1.0]])

            grid_image = grids.CoordinateGrid(regular_grid_coords)

            deflections = grid_image.deflection_grid_for_galaxies(galaxies=[galaxy_mass_sis])

            grid_traced = grid_image.ray_tracing_grid_for_deflections(deflections)

            assert grid_traced[0] == pytest.approx(np.array([1.0 - 0.707, 1.0 - 0.707]), 1e-2)
            assert grid_traced[1] == pytest.approx(np.array([-1.0 + 0.707, -1.0 + 0.707]), 1e-2)

        def test_three_identical_lenses__deflection_angles_triple(self, galaxy_mass_sis):
            regular_grid_coords = np.array([[1.0, 1.0]])

            grid_image = grids.CoordinateGrid(regular_grid_coords)

            deflections = grid_image.deflection_grid_for_galaxies(
                galaxies=[galaxy_mass_sis, galaxy_mass_sis, galaxy_mass_sis])

            grid_traced = grid_image.ray_tracing_grid_for_deflections(deflections)

            assert grid_traced == pytest.approx(np.array([[1.0 - 3.0 * 0.707, 1.0 - 3.0 * 0.707]]), 1e-3)

        def test_one_lens_with_three_identical_mass_profiles__deflection_angles_triple(self, lens_sis_x3):
            regular_grid_coords = np.array([[1.0, 1.0]])

            grid_image = grids.CoordinateGrid(regular_grid_coords)

            deflections = grid_image.deflection_grid_for_galaxies(galaxies=[lens_sis_x3])

            grid_traced = grid_image.ray_tracing_grid_for_deflections(deflections)

            assert grid_traced == pytest.approx(np.array([[1.0 - 3.0 * 0.707, 1.0 - 3.0 * 0.707]]), 1e-3)

@pytest.fixture(name="grid_image_sub")
def make_grid_image_sub():
    mask = np.array([[True, True, True],
                     [True, False, True],
                     [True, True, True]])

    mask = msk.Mask(array=mask, pixel_scale=3.0)
    return mask.sub_coordinate_grid_with_size(size=2)

class TestSubCoordinateGrid(object):

    class TestConstructor:

        def test__simple_grid_input__sets_up_grid_correctly_in_attributes(self):

            sub_grid_coords = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0],
                                        [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])

            grid_sub = grids.SubCoordinateGrid(grid_coords=sub_grid_coords, sub_grid_size=2)

            assert grid_sub.sub_grid_size == 2

            assert grid_sub[0] == pytest.approx(np.array([1.0, 1.0]), 1e-2)
            assert grid_sub[1] == pytest.approx(np.array([1.0, 1.0]), 1e-2)
            assert grid_sub[2] == pytest.approx(np.array([1.0, 1.0]), 1e-2)
            assert grid_sub[3] == pytest.approx(np.array([1.0, 1.0]), 1e-2)
            assert grid_sub[4] == pytest.approx(np.array([2.0, 2.0]), 1e-2)
            assert grid_sub[5] == pytest.approx(np.array([2.0, 2.0]), 1e-2)
            assert grid_sub[6] == pytest.approx(np.array([2.0, 2.0]), 1e-2)
            assert grid_sub[7] == pytest.approx(np.array([2.0, 2.0]), 1e-2)

    class TestIntensitiesViaGrid:

        def test__no_galaxies__intensities_returned_as_0s(self, galaxy_no_profiles):

            sub_grid_coords = np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0],
                                        [1.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

            sub_to_image = np.array([0, 0, 0, 0, 1, 1, 1, 1])

            grid_sub = grids.SubCoordinateGrid(sub_grid_coords, sub_grid_size=2)
            grid_mapping = grids.GridMapping(image_shape=(3,3), image_pixels=2, data_to_image=np.array([[0,0], [0,1]]),
                                             sub_grid_size=2, sub_to_image=sub_to_image)

            intensities = grid_sub.intensities_via_grid(galaxies=[galaxy_no_profiles], mapping=grid_mapping)

            assert intensities[0] == 0.0
            assert intensities[1] == 0.0

        def test__galaxy_light_sersic__intensities_returned_as_correct_values(self, galaxy_light_sersic):

            sub_grid_coords = np.array([[1.0, 1.0], [0.0, 1.0], [0.0, -1.0], [0.0, -1.0],
                                        [1.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.0, -1.0]])

            sub_to_image = np.array([0, 0, 0, 0, 1, 1, 1, 1])

            grid_sub = grids.SubCoordinateGrid(sub_grid_coords, sub_grid_size=2)
            grid_mapping = grids.GridMapping(image_shape=(3,3), image_pixels=2, data_to_image=np.array([[0,0], [0,1]]),
                                             sub_grid_size=2, sub_to_image=sub_to_image)


            intensity_00 = galaxy_light_sersic.intensity_at_coordinates(sub_grid_coords[0])
            intensity_01 = galaxy_light_sersic.intensity_at_coordinates(sub_grid_coords[1])
            intensity_02 = galaxy_light_sersic.intensity_at_coordinates(sub_grid_coords[2])
            intensity_03 = galaxy_light_sersic.intensity_at_coordinates(sub_grid_coords[3])
            intensity_10 = galaxy_light_sersic.intensity_at_coordinates(sub_grid_coords[4])
            intensity_11 = galaxy_light_sersic.intensity_at_coordinates(sub_grid_coords[5])
            intensity_12 = galaxy_light_sersic.intensity_at_coordinates(sub_grid_coords[6])
            intensity_13 = galaxy_light_sersic.intensity_at_coordinates(sub_grid_coords[7])

            intensity_0 = (intensity_00 + intensity_01 + intensity_02 + intensity_03) / 4.0
            intensity_1 = (intensity_10 + intensity_11 + intensity_12 + intensity_13) / 4.0

            intensities = grid_sub.intensities_via_grid(galaxies=[galaxy_light_sersic], mapping=grid_mapping)

            assert intensities[0] == intensity_0
            assert intensities[1] == intensity_1

        def test__galaxy_light_sersic_x3__deflections_tripled_from_above(self, galaxy_light_sersic):

            sub_grid_coords = np.array([[1.0, 1.0], [0.0, 1.0], [0.0, -1.0], [0.0, -1.0],
                                        [1.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.0, -1.0]])

            sub_to_image = np.array([0, 0, 0, 0, 1, 1, 1, 1])

            grid_sub = grids.SubCoordinateGrid(sub_grid_coords, sub_grid_size=2)
            grid_mapping = grids.GridMapping(image_shape=(3,3), image_pixels=2, data_to_image=np.array([[0,0], [0,1]]),
                                             sub_grid_size=2, sub_to_image=sub_to_image)


            intensity_00 = galaxy_light_sersic.intensity_at_coordinates(sub_grid_coords[0])
            intensity_01 = galaxy_light_sersic.intensity_at_coordinates(sub_grid_coords[1])
            intensity_02 = galaxy_light_sersic.intensity_at_coordinates(sub_grid_coords[2])
            intensity_03 = galaxy_light_sersic.intensity_at_coordinates(sub_grid_coords[3])
            intensity_10 = galaxy_light_sersic.intensity_at_coordinates(sub_grid_coords[4])
            intensity_11 = galaxy_light_sersic.intensity_at_coordinates(sub_grid_coords[5])
            intensity_12 = galaxy_light_sersic.intensity_at_coordinates(sub_grid_coords[6])
            intensity_13 = galaxy_light_sersic.intensity_at_coordinates(sub_grid_coords[7])

            intensity_0 = (intensity_00 + intensity_01 + intensity_02 + intensity_03) / 4.0
            intensity_1 = (intensity_10 + intensity_11 + intensity_12 + intensity_13) / 4.0

            intensities = grid_sub.intensities_via_grid(galaxies=[galaxy_light_sersic, galaxy_light_sersic,
                                                                  galaxy_light_sersic], mapping=grid_mapping)

            assert intensities[0] == pytest.approx(3.0 * intensity_0, 1e-3)
            assert intensities[1] == pytest.approx(3.0 * intensity_1, 1e-3)

    class TestDeflectionsGridForGalaxies:

        def test__simple_sis_model__deflection_angles(self, galaxy_mass_sis):

            sub_grid_coords = np.array([[1.0, 1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, -1.0],
                                        [1.0, 1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, -1.0]])

            grid_sub = grids.SubCoordinateGrid(sub_grid_coords, sub_grid_size=2)

            deflections = grid_sub.deflection_grid_for_galaxies(galaxies=[galaxy_mass_sis])

            assert deflections[0] == pytest.approx(np.array([0.707, 0.707]), 1e-2)
            assert deflections[1] == pytest.approx(np.array([0.707, 0.707]), 1e-2)
            assert deflections[2] == pytest.approx(np.array([-0.707, -0.707]), 1e-2)
            assert deflections[3] == pytest.approx(np.array([-0.707, -0.707]), 1e-2)
            assert deflections[4] == pytest.approx(np.array([0.707, 0.707]), 1e-2)
            assert deflections[5] == pytest.approx(np.array([0.707, 0.707]), 1e-2)
            assert deflections[6] == pytest.approx(np.array([-0.707, -0.707]), 1e-2)
            assert deflections[7] == pytest.approx(np.array([-0.707, -0.707]), 1e-2)

            assert deflections.sub_grid_size == 2

        def test_three_identical_lenses__deflection_angles_triple(self, galaxy_mass_sis):

            sub_grid_coords = np.array([[1.0, 1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, -1.0],
                                        [1.0, 1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, -1.0]])

            sub_to_image = np.array([0, 0, 0, 0, 1, 1, 1, 1])

            grid_sub = grids.SubCoordinateGrid(sub_grid_coords, sub_grid_size=2)

            deflections = grid_sub.deflection_grid_for_galaxies(
                galaxies=[galaxy_mass_sis, galaxy_mass_sis, galaxy_mass_sis])

            assert deflections[0] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]), 1e-2)
            assert deflections[1] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]), 1e-2)
            assert deflections[2] == pytest.approx(np.array([-3.0 * 0.707, -3.0 * 0.707]), 1e-2)
            assert deflections[3] == pytest.approx(np.array([-3.0 * 0.707, -3.0 * 0.707]), 1e-2)
            assert deflections[4] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]), 1e-2)
            assert deflections[5] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]), 1e-2)
            assert deflections[6] == pytest.approx(np.array([-3.0 * 0.707, -3.0 * 0.707]), 1e-2)
            assert deflections[7] == pytest.approx(np.array([-3.0 * 0.707, -3.0 * 0.707]), 1e-2)

            assert deflections.sub_grid_size == 2

        def test_one_lens_with_three_identical_mass_profiles__deflection_angles_triple(self, lens_sis_x3):

            sub_grid_coords = np.array([[1.0, 1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, -1.0],
                                        [1.0, 1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, -1.0]])

            sub_to_image = np.array([0, 0, 0, 0, 1, 1, 1, 1])

            grid_sub = grids.SubCoordinateGrid(sub_grid_coords, sub_grid_size=2)

            deflections = grid_sub.deflection_grid_for_galaxies(galaxies=[lens_sis_x3])

            assert deflections[0] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]), 1e-2)
            assert deflections[1] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]), 1e-2)
            assert deflections[2] == pytest.approx(np.array([-3.0 * 0.707, -3.0 * 0.707]), 1e-2)
            assert deflections[3] == pytest.approx(np.array([-3.0 * 0.707, -3.0 * 0.707]), 1e-2)
            assert deflections[4] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]), 1e-2)
            assert deflections[5] == pytest.approx(np.array([3.0 * 0.707, 3.0 * 0.707]), 1e-2)
            assert deflections[6] == pytest.approx(np.array([-3.0 * 0.707, -3.0 * 0.707]), 1e-2)
            assert deflections[7] == pytest.approx(np.array([-3.0 * 0.707, -3.0 * 0.707]), 1e-2)

            assert deflections.sub_grid_size == 2

    class TestSetupTracedGrid:

        def test__simple_sis_model__deflection_angles(self, galaxy_mass_sis):

            sub_grid_coords = np.array([[1.0, 1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, -1.0],
                                        [1.0, 1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, -1.0]])

            sub_to_image = np.array([0, 0, 0, 0, 1, 1, 1, 1])

            grid_sub = grids.SubCoordinateGrid(sub_grid_coords, sub_grid_size=2)

            deflections = grid_sub.deflection_grid_for_galaxies(galaxies=[galaxy_mass_sis])

            grid_traced = grid_sub.ray_tracing_grid_for_deflections(deflections)

            assert grid_traced[0] == pytest.approx(np.array([1.0 - 0.707, 1.0 - 0.707]), 1e-2)
            assert grid_traced[1] == pytest.approx(np.array([1.0 - 0.707, 1.0 - 0.707]), 1e-2)
            assert grid_traced[2] == pytest.approx(np.array([-1.0 + 0.707, -1.0 + 0.707]), 1e-2)
            assert grid_traced[3] == pytest.approx(np.array([-1.0 + 0.707, -1.0 + 0.707]), 1e-2)
            assert grid_traced[4] == pytest.approx(np.array([1.0 - 0.707, 1.0 - 0.707]), 1e-2)
            assert grid_traced[5] == pytest.approx(np.array([1.0 - 0.707, 1.0 - 0.707]), 1e-2)
            assert grid_traced[6] == pytest.approx(np.array([-1.0 + 0.707, -1.0 + 0.707]), 1e-2)
            assert grid_traced[7] == pytest.approx(np.array([-1.0 + 0.707, -1.0 + 0.707]), 1e-2)

            assert grid_traced.sub_grid_size == 2

        def test_three_identical_lenses__deflection_angles_triple(self, galaxy_mass_sis):

            sub_grid_coords = np.array([[1.0, 1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, -1.0],
                                        [1.0, 1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, -1.0]])

            sub_to_image = np.array([0, 0, 0, 0, 1, 1, 1, 1])

            grid_sub = grids.SubCoordinateGrid(sub_grid_coords, sub_grid_size=2)

            deflections = grid_sub.deflection_grid_for_galaxies(
                galaxies=[galaxy_mass_sis, galaxy_mass_sis, galaxy_mass_sis])

            grid_traced = grid_sub.ray_tracing_grid_for_deflections(deflections)

            assert grid_traced[0] == pytest.approx(np.array([1.0 - 3.0 * 0.707, 1.0 - 3.0 * 0.707]), 1e-2)
            assert grid_traced[1] == pytest.approx(np.array([1.0 - 3.0 * 0.707, 1.0 - 3.0 * 0.707]), 1e-2)
            assert grid_traced[2] == pytest.approx(np.array([-1.0 + 3.0 * 0.707, -1.0 + 3.0 * 0.707]), 1e-2)
            assert grid_traced[3] == pytest.approx(np.array([-1.0 + 3.0 * 0.707, -1.0 + 3.0 * 0.707]), 1e-2)
            assert grid_traced[4] == pytest.approx(np.array([1.0 - 3.0 * 0.707, 1.0 - 3.0 * 0.707]), 1e-2)
            assert grid_traced[5] == pytest.approx(np.array([1.0 - 3.0 * 0.707, 1.0 - 3.0 * 0.707]), 1e-2)
            assert grid_traced[6] == pytest.approx(np.array([-1.0 + 3.0 * 0.707, -1.0 + 3.0 * 0.707]), 1e-2)
            assert grid_traced[7] == pytest.approx(np.array([-1.0 + 3.0 * 0.707, -1.0 + 3.0 * 0.707]), 1e-2)

            assert grid_traced.sub_grid_size == 2

        def test_one_lens_with_three_identical_mass_profiles__deflection_angles_triple(self, lens_sis_x3):

            sub_grid_coords = np.array([[1.0, 1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, -1.0],
                                        [1.0, 1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, -1.0]])

            sub_to_image = np.array([0, 0, 0, 0, 1, 1, 1, 1])

            grid_sub = grids.SubCoordinateGrid(sub_grid_coords, sub_grid_size=2)

            deflections = grid_sub.deflection_grid_for_galaxies(galaxies=[lens_sis_x3])

            grid_traced = grid_sub.ray_tracing_grid_for_deflections(deflections)

            assert grid_traced[0] == pytest.approx(np.array([1.0 - 3.0 * 0.707, 1.0 - 3.0 * 0.707]), 1e-2)
            assert grid_traced[1] == pytest.approx(np.array([1.0 - 3.0 * 0.707, 1.0 - 3.0 * 0.707]), 1e-2)
            assert grid_traced[2] == pytest.approx(np.array([-1.0 + 3.0 * 0.707, -1.0 + 3.0 * 0.707]), 1e-2)
            assert grid_traced[3] == pytest.approx(np.array([-1.0 + 3.0 * 0.707, -1.0 + 3.0 * 0.707]), 1e-2)
            assert grid_traced[4] == pytest.approx(np.array([1.0 - 3.0 * 0.707, 1.0 - 3.0 * 0.707]), 1e-2)
            assert grid_traced[5] == pytest.approx(np.array([1.0 - 3.0 * 0.707, 1.0 - 3.0 * 0.707]), 1e-2)
            assert grid_traced[6] == pytest.approx(np.array([-1.0 + 3.0 * 0.707, -1.0 + 3.0 * 0.707]), 1e-2)
            assert grid_traced[7] == pytest.approx(np.array([-1.0 + 3.0 * 0.707, -1.0 + 3.0 * 0.707]), 1e-2)

            assert deflections.sub_grid_size == 2

    class TestDeflectionsOnGrid:

        def test__no_galaxies__deflections_returned_as_0s(self, galaxy_no_profiles):

            sub_grid_coords = np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0],
                                        [1.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

            grid_sub = grids.SubCoordinateGrid(sub_grid_coords, sub_grid_size=2)

            deflections = grid_sub.deflections_on_grid(galaxies=[galaxy_no_profiles])

            assert deflections[0] == pytest.approx(np.array([0.0, 0.0]), 1e-2)
            assert deflections[1] == pytest.approx(np.array([0.0, 0.0]), 1e-2)
            assert deflections[2] == pytest.approx(np.array([0.0, 0.0]), 1e-2)
            assert deflections[3] == pytest.approx(np.array([0.0, 0.0]), 1e-2)
            assert deflections[4] == pytest.approx(np.array([0.0, 0.0]), 1e-2)
            assert deflections[5] == pytest.approx(np.array([0.0, 0.0]), 1e-2)
            assert deflections[6] == pytest.approx(np.array([0.0, 0.0]), 1e-2)
            assert deflections[7] == pytest.approx(np.array([0.0, 0.0]), 1e-2)

        def test__galaxy_sis_mass__deflections_returned_as_correct_values(self, galaxy_mass_sis):

            sub_grid_coords = np.array([[1.0, 1.0], [0.0, 1.0], [0.0, -1.0], [0.0, -1.0],
                                        [1.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.0, -1.0]])

            grid_sub = grids.SubCoordinateGrid(sub_grid_coords, sub_grid_size=2)

            deflections = grid_sub.deflections_on_grid(galaxies=[galaxy_mass_sis])

            assert deflections[0] == pytest.approx(np.array([0.707, 0.707]), 1e-2)
            assert deflections[1] == pytest.approx(np.array([0.0, 1.0]), 1e-2)
            assert deflections[2] == pytest.approx(np.array([0.0, -1.0]), 1e-2)
            assert deflections[3] == pytest.approx(np.array([0.0, -1.0]), 1e-2)
            assert deflections[4] == pytest.approx(np.array([0.707, 0.707]), 1e-2)
            assert deflections[5] == pytest.approx(np.array([-1.0, 0.0]), 1e-2)
            assert deflections[6] == pytest.approx(np.array([0.0, -1.0]), 1e-2)
            assert deflections[7] == pytest.approx(np.array([0.0, -1.0]), 1e-2)

        def test__galaxy_sis_mass_x3__deflections_tripled_from_above(self, galaxy_mass_sis):

            sub_grid_coords = np.array([[1.0, 1.0], [0.0, 1.0], [0.0, -1.0], [0.0, -1.0],
                                        [1.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.0, -1.0]])

            grid_sub = grids.SubCoordinateGrid(sub_grid_coords, sub_grid_size=2)

            deflections = grid_sub.deflections_on_grid(galaxies=[galaxy_mass_sis, galaxy_mass_sis, galaxy_mass_sis])

            assert deflections[0] == pytest.approx(3.0 * np.array([0.707, 0.707]), 1e-2)
            assert deflections[1] == pytest.approx(3.0 * np.array([0.0, 1.0]), 1e-2)
            assert deflections[2] == pytest.approx(3.0 * np.array([0.0, -1.0]), 1e-2)
            assert deflections[3] == pytest.approx(3.0 * np.array([0.0, -1.0]), 1e-2)
            assert deflections[4] == pytest.approx(3.0 * np.array([0.707, 0.707]), 1e-2)
            assert deflections[5] == pytest.approx(3.0 * np.array([-1.0, 0.0]), 1e-2)
            assert deflections[6] == pytest.approx(3.0 * np.array([0.0, -1.0]), 1e-2)
            assert deflections[7] == pytest.approx(3.0 * np.array([0.0, -1.0]), 1e-2)


class TestDataCollection(object):

    class TestConstructor:

        def test__all_grid_datas_entered__sets_up_attributes(self):

            grid_image = np.array([1, 2, 3])
            data_to_image = np.array([[0,1], [0,2], [0,3]])
            grid_image = grids.GridData(grid_image)

            grid_noise = np.array([4, 5, 6])
            grid_noise = grids.GridData(grid_noise)

            grid_exposure_time = np.array([7, 8, 9])
            grid_exposure_time = grids.GridData(grid_exposure_time)

            grid_collection = grids.DataCollection(image=grid_image, noise=grid_noise,
                                                   exposure_time=grid_exposure_time)

            assert (grid_collection.image == np.array([1, 2, 3])).all()
            assert (grid_collection.noise == np.array([4, 5, 6])).all()
            assert (grid_collection.exposure_time == np.array([7, 8, 9])).all()

    class TestFromMask:

        def test__cross_mask__all_data_setup_as_mask(self):

            mask = np.array([[True, False, True],
                             [False, False, False],
                             [True, False, True]])
            mask = msk.Mask(array=mask, pixel_scale=1.0)

            image = np.array([[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9]])
            image = img.Image(array=image, pixel_scale=1.0)

            noise = np.array([[2, 2, 2],
                              [5, 5, 5],
                              [8, 8, 8]])
            noise = img.ScaledArray(array=noise, pixel_scale=1.0)

            exposure_time = np.array([[1, 1, 1],
                                      [1, 1, 1],
                                      [1, 1, 1]])
            exposure_time = img.ScaledArray(array=exposure_time, pixel_scale=1.0)

            grid_collection = grids.DataCollection.from_mask(mask=mask, image=image, noise=noise,
                                                             exposure_time=exposure_time)

            assert (grid_collection.image == np.array([2, 4, 5, 6, 8])).all()
            assert (grid_collection.noise == np.array([2, 5, 5, 5, 8])).all()
            assert (grid_collection.exposure_time == np.array([1, 1, 1, 1, 1])).all()


class TestGridData(object):

    class TestConstructor:

        def test__simple_grid_input__sets_up_grid_correctly_in_attributes(self):

            grid_data = np.array([1, 2, 3])
            data_to_image = np.array([[0, 0], [0, 1], [0, 2]])

            grid_data = grids.GridData(grid_data)

            assert (grid_data[0] == np.array([1])).all()
            assert (grid_data[1] == np.array([2])).all()
            assert (grid_data[2] == np.array([3])).all()

    class TestFromMask:

        def test__simple_grid_input_via_mask(self):

            data = np.array([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])

            mask = np.array([[True, False, True],
                             [False, False, False],
                             [True, False, True]])

            mask = msk.Mask(mask, pixel_scale=3.0)

            grid_data = grids.GridData.from_mask(data, mask)

            assert (grid_data[0] == np.array([2])).all()
            assert (grid_data[1] == np.array([4])).all()
            assert (grid_data[2] == np.array([5])).all()
            assert (grid_data[3] == np.array([6])).all()
            assert (grid_data[4] == np.array([8])).all()


class TestGridMapping(object):

    class TestFromMask:

        def test__setup_mappings_using_mask(self):

            mask = np.array([[True, False, True],
                             [False, False, False],
                             [True, False, True]])

            mask = msk.Mask(mask, pixel_scale=3.0)

            grid_mapping = grids.GridMapping.from_mask(mask, sub_grid_size=2)

            assert grid_mapping.image_shape == (3, 3)
            assert grid_mapping.image_pixels == 5

            assert (grid_mapping.data_to_image[0] == np.array([0, 1])).all()
            assert (grid_mapping.data_to_image[1] == np.array([1, 0])).all()
            assert (grid_mapping.data_to_image[2] == np.array([1, 1])).all()
            assert (grid_mapping.data_to_image[3] == np.array([1, 2])).all()
            assert (grid_mapping.data_to_image[4] == np.array([2, 1])).all()

            assert grid_mapping.sub_grid_size == 2
            assert grid_mapping.sub_grid_fraction == (1.0/4.0)

            assert (grid_mapping.sub_to_image == np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])).all()

        def test__cluster_is_setup_with_cluster_sub_grid_size(self):

            mask = np.array([[True, True, True],
                             [True, False, True],
                             [True, True, True]])

            mask = msk.Mask(array=mask, pixel_scale=3.0)

            cluster_to_image, image_to_cluster = mask.compute_grid_mapper_sparse(sparse_grid_size=1)

            mapping = grids.GridMapping.from_mask(mask, sub_grid_size=2, cluster_grid_size=1)

            assert (cluster_to_image == mapping.cluster.cluster_to_image).all()
            assert (image_to_cluster == mapping.cluster.image_to_cluster).all()

    class TestMapDataTo2d:

        def test__3x3_dataset_in_2d__mask_is_all_false__maps_back_to_original_data(self):

            data = np.array([[0, 1, 2],
                             [3, 4, 5],
                             [6, 7, 8]])

            mask = np.array([[False, False, False],
                             [False, False, False],
                             [False, False, False]])

            data_to_image = np.array([[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1], [2,2]])

            mask = msk.Mask(mask, pixel_scale=3.0)

            grid_data = grids.GridData.from_mask(mask=mask, data=data)

            grid_mapping = grids.GridMapping(image_shape=(3,3), image_pixels=9, data_to_image=data_to_image,
                                             sub_grid_size=1, sub_to_image=np.array([0]))

            data_2d = grid_mapping.map_to_2d(grid_data=grid_data)

            assert (data == data_2d).all()

            assert (data == np.array([[0, 1, 2],
                                      [3, 4, 5],
                                      [6, 7, 8]])).all()

        def test__3x3_dataset_in_2d__mask_has_trues_in_it__zeros_where_mask_is_true(self):

            data = np.array([[0, 1, 2],
                             [3, 4, 5],
                             [6, 7, 8]])

            mask = np.array([[True, False, True],
                             [False, False, False],
                             [True, False, True]])

            data_to_image = np.array([[0,1], [1,0], [1,1], [1,2], [2,1]])

            mask = msk.Mask(mask, pixel_scale=3.0)

            grid_data = grids.GridData.from_mask(mask=mask, data=data)

            grid_mapping = grids.GridMapping(image_shape=(3,3), image_pixels=1, data_to_image=data_to_image,
                                             sub_grid_size=5, sub_to_image=np.array([0]))

            data_2d = grid_mapping.map_to_2d(grid_data=grid_data)

            assert (data_2d == np.array([[0, 1, 0],
                                        [3, 4, 5],
                                        [0, 7, 0]])).all()

    class TestMapDataToImageGrid:

        def test__2x2_sub_grid__image_is_1_pixel(self):

            data_to_image = np.array([[0,0]])
            sub_to_image = np.array([0, 0, 0, 0])

            grid_sub = grids.GridMapping(image_shape=(3,3), image_pixels=1, data_to_image=data_to_image,
                                         sub_grid_size=2, sub_to_image=sub_to_image)

            image_data = grid_sub.map_data_sub_to_image(data=np.array([1.0, 2.0, 3.0, 4.0]))

            assert image_data == (1.0+2.0+3.0+4.0)/4.0

        def test__2x2_sub_grid__image_is_4_pixels(self):

            data_to_image = np.array([[0,0], [0,1], [1,0], [1,1]])

            sub_to_image = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])

            grid_sub = grids.GridMapping(image_shape=(4,4), image_pixels=4, data_to_image=data_to_image,
                                         sub_grid_size=2, sub_to_image=sub_to_image)

            image_data = grid_sub.map_data_sub_to_image(data=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                                                                       9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]))

            assert image_data[0] == (1.0+2.0+3.0+4.0)/4.0
            assert image_data[1] == (5.0+6.0+7.0+8.0)/4.0
            assert image_data[2] == (9.0+10.0+11.0+12.0)/4.0
            assert image_data[3] == (13.0+14.0+15.0+16.0)/4.0

        def test__3x3_sub_grid__image_is_6_pixels(self):

            data_to_image = np.array([[0,0], [0,1], [1,0], [1,1], [2,0], [2,1]])

            coords = np.array([[1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [0.0, 0.0], [0.0, 0.0],
                               [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [0.0, 0.0], [0.0, 0.0],
                               [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [0.0, 0.0], [0.0, 0.0],
                               [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [0.0, 0.0], [0.0, 0.0],
                               [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [0.0, 0.0], [0.0, 0.0],
                               [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [0.0, 0.0], [0.0, 0.0]])

            sub_to_image = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     2, 2, 2, 2, 2, 2, 2, 2, 2,
                                     3, 3, 3, 3, 3, 3, 3, 3, 3,
                                     4, 4, 4, 4, 4, 4, 4, 4, 4,
                                     5, 5, 5, 5, 5, 5, 5, 5, 5])

            grid_sub = grids.GridMapping(image_shape=(4,4), image_pixels=6, data_to_image=data_to_image,
                                               sub_grid_size=3, sub_to_image=sub_to_image)

            image_data = grid_sub.map_data_sub_to_image(data=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                                                       2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                                                       3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                                                                       4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                                                       5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                                                                       6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]))

            assert image_data[0] == (1.0*9.0)/9.0
            assert image_data[1] == (2.0*9.0)/9.0
            assert image_data[2] == (3.0*9.0)/9.0
            assert image_data[3] == (4.0*9.0)/9.0
            assert image_data[4] == (5.0*9.0)/9.0
            assert image_data[5] == (6.0*9.0)/9.0


class TestGridClusterPixelization(object):

    class TestConstructor:

        def test__simple_mapper_input__sets_up_grid_in_attributes(self):

            cluster_to_image = np.array([1, 2, 3, 5])
            image_to_cluster = np.array([6, 7, 2, 3])

            cluster_pix = grids.GridClusterPixelization(cluster_to_image, image_to_cluster)

            assert (cluster_pix.cluster_to_image == np.array([1, 2, 3, 5])).all()
            assert (cluster_pix.image_to_cluster == np.array([6, 7, 2, 3])).all()

    class TestFromMask:

        def test__simple_constructor__compare_to_manual_setup_via_mask(self):

            mask = np.array([[True, True, True],
                             [True, False, True],
                             [True, True, True]])

            mask = msk.Mask(array=mask, pixel_scale=3.0)

            cluster_to_image, image_to_cluster = mask.compute_grid_mapper_sparse(sparse_grid_size=1)

            cluster_pix = grids.GridClusterPixelization(cluster_to_image, image_to_cluster)

            cluster_pix_from_mask = grids.GridClusterPixelization.from_mask(mask, cluster_grid_size=1)

            assert (cluster_pix.cluster_to_image == cluster_pix_from_mask.cluster_to_image).all()
            assert (cluster_pix.image_to_cluster == cluster_pix_from_mask.image_to_cluster).all()


class TestGridBorder(object):
    class TestCoordinatesAngleFromX(object):

        def test__angle_is_zero__angles_follow_trig(self):
            coordinates = np.array([1.0, 0.0])

            border = grids.GridBorder(border_pixels=np.array([0]), polynomial_degree=3)

            theta_from_x = border.coordinates_angle_from_x(coordinates)

            assert theta_from_x == 0.0

        def test__angle_is_forty_five__angles_follow_trig(self):
            coordinates = np.array([1.0, 1.0])

            border = grids.GridBorder(border_pixels=np.array([0]), polynomial_degree=3)

            theta_from_x = border.coordinates_angle_from_x(coordinates)

            assert theta_from_x == 45.0

        def test__angle_is_sixty__angles_follow_trig(self):
            coordinates = np.array([1.0, 1.7320])

            border = grids.GridBorder(border_pixels=np.array([0]), polynomial_degree=3)

            theta_from_x = border.coordinates_angle_from_x(coordinates)

            assert theta_from_x == pytest.approx(60.0, 1e-3)

        def test__top_left_quandrant__angle_goes_above_90(self):
            coordinates = np.array([-1.0, 1.0])

            border = grids.GridBorder(border_pixels=np.array([0]), polynomial_degree=3)

            theta_from_x = border.coordinates_angle_from_x(coordinates)

            assert theta_from_x == 135.0

        def test__bottom_left_quandrant__angle_continues_above_180(self):
            coordinates = np.array([-1.0, -1.0])

            border = grids.GridBorder(border_pixels=np.array([1]), polynomial_degree=3)

            theta_from_x = border.coordinates_angle_from_x(coordinates)

            assert theta_from_x == 225.0

        def test__bottom_right_quandrant__angle_flips_back_to_above_90(self):
            coordinates = np.array([1.0, -1.0])

            border = grids.GridBorder(border_pixels=np.array([0]), polynomial_degree=3)

            theta_from_x = border.coordinates_angle_from_x(coordinates)

            assert theta_from_x == 315.0

        def test__include_source_plane_centre__angle_takes_into_accounts(self):
            coordinates = np.array([2.0, 2.0])

            border = grids.GridBorder(border_pixels=np.array([0]), polynomial_degree=3, centre=(1.0, 1.0))

            theta_from_x = border.coordinates_angle_from_x(coordinates)

            assert theta_from_x == 45.0

    class TestThetasAndRadii:

        def test__four_coordinates_in_circle__all_in_border__correct_radii_and_thetas(self):
            coordinates = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])

            border = grids.GridBorder(border_pixels=np.arange(4), polynomial_degree=3)
            border.polynomial_fit_to_border(coordinates)

            assert border.radii == [1.0, 1.0, 1.0, 1.0]
            assert border.thetas == [0.0, 90.0, 180.0, 270.0]

        def test__other_thetas_radii(self):
            coordinates = np.array([[2.0, 0.0], [2.0, 2.0], [-1.0, -1.0], [0.0, -3.0]])

            border = grids.GridBorder(border_pixels=np.arange(4), polynomial_degree=3)
            border.polynomial_fit_to_border(coordinates)

            assert border.radii == [2.0, 2.0 * np.sqrt(2), np.sqrt(2.0), 3.0]
            assert border.thetas == [0.0, 45.0, 225.0, 270.0]

        def test__border_centre_offset__coordinates_same_r_and_theta_shifted(self):
            coordinates = np.array([[2.0, 1.0], [1.0, 2.0], [0.0, 1.0], [1.0, 0.0]])

            border = grids.GridBorder(border_pixels=np.arange(4), polynomial_degree=3, centre=(1.0, 1.0))
            border.polynomial_fit_to_border(coordinates)

            assert border.radii == [1.0, 1.0, 1.0, 1.0]
            assert border.thetas == [0.0, 90.0, 180.0, 270.0]

    class TestBorderPolynomial(object):

        def test__four_coordinates_in_circle__thetas_at_radius_are_each_coordinates_radius(self):
            coordinates = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])

            border = grids.GridBorder(border_pixels=np.arange(4), polynomial_degree=3)
            border.polynomial_fit_to_border(coordinates)

            assert border.radius_at_theta(theta=0.0) == pytest.approx(1.0, 1e-3)
            assert border.radius_at_theta(theta=90.0) == pytest.approx(1.0, 1e-3)
            assert border.radius_at_theta(theta=180.0) == pytest.approx(1.0, 1e-3)
            assert border.radius_at_theta(theta=270.0) == pytest.approx(1.0, 1e-3)

        def test__eight_coordinates_in_circle__thetas_at_each_coordinates_are_the_radius(self):
            coordinates = np.array([[1.0, 0.0], [0.5 * np.sqrt(2), 0.5 * np.sqrt(2)],
                                    [0.0, 1.0], [-0.5 * np.sqrt(2), 0.5 * np.sqrt(2)],
                                    [-1.0, 0.0], [-0.5 * np.sqrt(2), -0.5 * np.sqrt(2)],
                                    [0.0, -1.0], [0.5 * np.sqrt(2), -0.5 * np.sqrt(2)]])

            border = grids.GridBorder(border_pixels=
                                      np.arange(8), polynomial_degree=3)
            border.polynomial_fit_to_border(coordinates)

            assert border.radius_at_theta(theta=0.0) == pytest.approx(1.0, 1e-3)
            assert border.radius_at_theta(theta=45.0) == pytest.approx(1.0, 1e-3)
            assert border.radius_at_theta(theta=90.0) == pytest.approx(1.0, 1e-3)
            assert border.radius_at_theta(theta=135.0) == pytest.approx(1.0, 1e-3)
            assert border.radius_at_theta(theta=180.0) == pytest.approx(1.0, 1e-3)
            assert border.radius_at_theta(theta=225.0) == pytest.approx(1.0, 1e-3)
            assert border.radius_at_theta(theta=270.0) == pytest.approx(1.0, 1e-3)
            assert border.radius_at_theta(theta=315.0) == pytest.approx(1.0, 1e-3)

    class TestMoveFactors(object):

        def test__inside_border__move_factor_is_1(self):
            coordinates = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])

            border = grids.GridBorder(border_pixels=np.arange(4), polynomial_degree=3)
            border.polynomial_fit_to_border(coordinates)

            assert border.move_factor(coordinate=(0.5, 0.0)) == 1.0
            assert border.move_factor(coordinate=(-0.5, 0.0)) == 1.0
            assert border.move_factor(coordinate=(0.25, 0.25)) == 1.0
            assert border.move_factor(coordinate=(0.0, 0.0)) == 1.0

        def test__outside_border_double_its_radius__move_factor_is_05(self):
            coordinates = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])

            border = grids.GridBorder(border_pixels=np.arange(4), polynomial_degree=3)
            border.polynomial_fit_to_border(coordinates)

            assert border.move_factor(coordinate=(2.0, 0.0)) == pytest.approx(0.5, 1e-3)
            assert border.move_factor(coordinate=(0.0, 2.0)) == pytest.approx(0.5, 1e-3)
            assert border.move_factor(coordinate=(-2.0, 0.0)) == pytest.approx(0.5, 1e-3)
            assert border.move_factor(coordinate=(0.0, -2.0)) == pytest.approx(0.5, 1e-3)

        def test__outside_border_double_its_radius_and_offset__move_factor_is_05(self):
            coordinates = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])

            border = grids.GridBorder(border_pixels=np.arange(4), polynomial_degree=3)
            border.polynomial_fit_to_border(coordinates)

            assert border.move_factor(coordinate=(2.0, 0.0)) == pytest.approx(0.5, 1e-3)
            assert border.move_factor(coordinate=(0.0, 2.0)) == pytest.approx(0.5, 1e-3)
            assert border.move_factor(coordinate=(0.0, 2.0)) == pytest.approx(0.5, 1e-3)
            assert border.move_factor(coordinate=(2.0, 0.0)) == pytest.approx(0.5, 1e-3)

        def test__outside_border_as_above__but_shift_for_source_plane_centre(self):
            coordinates = np.array([[2.0, 1.0], [1.0, 2.0], [0.0, 1.0], [1.0, 0.0]])

            border = grids.GridBorder(border_pixels=np.arange(4), polynomial_degree=3, centre=(1.0, 1.0))
            border.polynomial_fit_to_border(coordinates)

            assert border.move_factor(coordinate=(3.0, 1.0)) == pytest.approx(0.5, 1e-3)
            assert border.move_factor(coordinate=(1.0, 3.0)) == pytest.approx(0.5, 1e-3)
            assert border.move_factor(coordinate=(1.0, 3.0)) == pytest.approx(0.5, 1e-3)
            assert border.move_factor(coordinate=(3.0, 1.0)) == pytest.approx(0.5, 1e-3)

    class TestRelocateCoordinates(object):

        def test__inside_border_no_relocations(self):
            thetas = np.linspace(0.0, 2.0 * np.pi, 32)
            coordinates = np.asarray(list(map(lambda x: (np.cos(x), np.sin(x)), thetas)))

            border = grids.GridBorder(border_pixels=np.arange(32), polynomial_degree=3)
            border.polynomial_fit_to_border(coordinates)

            assert border.relocated_coordinate(coordinate=np.array([0.1, 0.0])) == \
                   pytest.approx(np.array([0.1, 0.0]), 1e-3)

            assert border.relocated_coordinate(coordinate=np.array([-0.2, -0.3])) == \
                   pytest.approx(np.array([-0.2, -0.3]), 1e-3)

            assert border.relocated_coordinate(coordinate=np.array([0.5, 0.4])) == \
                   pytest.approx(np.array([0.5, 0.4]), 1e-3)

            assert border.relocated_coordinate(coordinate=np.array([0.7, -0.1])) == \
                   pytest.approx(np.array([0.7, -0.1]), 1e-3)

        def test__outside_border_simple_cases__relocates_to_source_border(self):
            thetas = np.linspace(0.0, 2.0 * np.pi, 32)
            coordinates = np.asarray(list(map(lambda x: (np.cos(x), np.sin(x)), thetas)))

            border = grids.GridBorder(border_pixels=np.arange(32), polynomial_degree=3)
            border.polynomial_fit_to_border(coordinates)

            assert border.relocated_coordinate(coordinate=np.array([2.5, 0.0])) == \
                   pytest.approx(np.array([1.0, 0.0]), 1e-3)

            assert border.relocated_coordinate(coordinate=np.array([0.0, 3.0])) == \
                   pytest.approx(np.array([0.0, 1.0]), 1e-3)

            assert border.relocated_coordinate(coordinate=np.array([-2.5, 0.0])) == \
                   pytest.approx(np.array([-1.0, 0.0]), 1e-3)

            assert border.relocated_coordinate(coordinate=np.array([-5.0, 5.0])) == \
                   pytest.approx(np.array([-0.707, 0.707]), 1e-3)

        def test__outside_border_simple_cases_2__relocates_to_source_border(self):
            thetas = np.linspace(0.0, 2.0 * np.pi, 16)
            coordinates = np.asarray(list(map(lambda x: (np.cos(x), np.sin(x)), thetas)))

            border = grids.GridBorder(border_pixels=np.arange(16), polynomial_degree=3)
            border.polynomial_fit_to_border(coordinates)

            assert border.relocated_coordinate(coordinate=(2.0, 0.0)) == pytest.approx((1.0, 0.0), 1e-3)

            assert border.relocated_coordinate(coordinate=(0.0, 2.0)) == pytest.approx((0.0, 1.0), 1e-3)

            assert border.relocated_coordinate(coordinate=(-2.0, 0.0)) == pytest.approx((-1.0, 0.0), 1e-3)

            assert border.relocated_coordinate(coordinate=(0.0, -1.0)) == pytest.approx((0.0, -1.0), 1e-3)

            assert border.relocated_coordinate(coordinate=(1.0, 1.0)) == \
                   pytest.approx((0.5 * np.sqrt(2), 0.5 * np.sqrt(2)), 1e-3)

            assert border.relocated_coordinate(coordinate=(-1.0, 1.0)) == \
                   pytest.approx((-0.5 * np.sqrt(2), 0.5 * np.sqrt(2)), 1e-3)

            assert border.relocated_coordinate(coordinate=(-1.0, -1.0)) == \
                   pytest.approx((-0.5 * np.sqrt(2), -0.5 * np.sqrt(2)), 1e-3)

            assert border.relocated_coordinate(coordinate=(1.0, -1.0)) == \
                   pytest.approx((0.5 * np.sqrt(2), -0.5 * np.sqrt(2)), 1e-3)

    class TestRelocateAllCoordinatesOutsideBorder(object):

        def test__coordinates_inside_border__no_relocations(self):
            thetas = np.linspace(0.0, 2.0 * np.pi, 16)
            circle = list(map(lambda x: (np.cos(x), np.sin(x)), thetas))

            coordinates = np.asarray(circle + [(0.1, 0.0), (0.1, 0.0), (0.0, 0.1), (-0.1, 0.0),
                                               (-0.1, 0.0), (-0.1, -0.0), (0.0, -0.1), (0.1, -0.0)])

            border_pixels = np.arange(16)
            border = grids.GridBorder(border_pixels, polynomial_degree=3)

            relocated_coordinates = border.relocate_coordinates_outside_border(coordinates)

            assert relocated_coordinates == pytest.approx(coordinates, 1e-3)

        def test__all_coordinates_inside_border_again__no_relocations(self):
            thetas = np.linspace(0.0, 2.0 * np.pi, 16)
            circle = list(map(lambda x: (np.cos(x), np.sin(x)), thetas))

            coordinates = np.asarray(circle + [(0.5, 0.0), (0.5, 0.5), (0.0, 0.5), (-0.5, 0.5),
                                               (-0.5, 0.0), (-0.5, -0.5), (0.0, -0.5), (0.5, -0.5)])

            border_pixels = np.arange(16)
            border = grids.GridBorder(border_pixels, polynomial_degree=3)

            relocated_coordinates = border.relocate_coordinates_outside_border(coordinates)

            assert relocated_coordinates == pytest.approx(coordinates, 1e-3)

        def test__6_coordinates_total__2_outside_border__relocate_to_source_border(self):
            coordinates = np.array([[1.0, 0.0], [20., 20.], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]])
            border_pixels = np.array([0, 2, 3, 4])

            border = grids.GridBorder(border_pixels, polynomial_degree=3)

            relocated_coordinates = border.relocate_coordinates_outside_border(coordinates)

            assert relocated_coordinates[0] == pytest.approx(coordinates[0], 1e-3)
            assert relocated_coordinates[1] == pytest.approx(np.array([0.7071, 0.7071]), 1e-3)
            assert relocated_coordinates[2] == pytest.approx(coordinates[2], 1e-3)
            assert relocated_coordinates[3] == pytest.approx(coordinates[3], 1e-3)
            assert relocated_coordinates[4] == pytest.approx(coordinates[4], 1e-3)
            assert relocated_coordinates[5] == pytest.approx(np.array([0.7071, 0.7071]), 1e-3)

        def test__24_coordinates_total__8_coordinates_outside_border__relocate_to_source_border(self):
            thetas = np.linspace(0.0, 2.0 * np.pi, 16)
            circle = list(map(lambda x: (np.cos(x), np.sin(x)), thetas))
            coordinates = np.asarray(circle + [(2.0, 0.0), (1.0, 1.0), (0.0, 2.0), (-1.0, 1.0),
                                               (-2.0, 0.0), (-1.0, -1.0), (0.0, -2.0), (1.0, -1.0)])

            border_pixels = np.arange(16)
            border = grids.GridBorder(border_pixels, polynomial_degree=3)

            relocated_coordinates = border.relocate_coordinates_outside_border(coordinates)

            assert relocated_coordinates[0:16] == pytest.approx(coordinates[0:16], 1e-3)
            assert relocated_coordinates[16] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert relocated_coordinates[17] == pytest.approx(np.array([0.5 * np.sqrt(2), 0.5 * np.sqrt(2)]),
                                                              1e-3)
            assert relocated_coordinates[18] == pytest.approx(np.array([0.0, 1.0]), 1e-3)
            assert relocated_coordinates[19] == pytest.approx(
                np.array([-0.5 * np.sqrt(2), 0.5 * np.sqrt(2)]), 1e-3)
            assert relocated_coordinates[20] == pytest.approx(np.array([-1.0, 0.0]), 1e-3)
            assert relocated_coordinates[21] == pytest.approx(
                np.array([-0.5 * np.sqrt(2), -0.5 * np.sqrt(2)]), 1e-3)
            assert relocated_coordinates[22] == pytest.approx(np.array([0.0, -1.0]), 1e-3)
            assert relocated_coordinates[23] == pytest.approx(
                np.array([0.5 * np.sqrt(2), -0.5 * np.sqrt(2)]), 1e-3)

        def test__24_coordinates_total__4_coordinates_outside_border__relates_to_source_border(self):
            thetas = np.linspace(0.0, 2.0 * np.pi, 16)
            circle = list(map(lambda x: (np.cos(x), np.sin(x)), thetas))
            coordinates = np.asarray(circle + [(0.5, 0.0), (0.5, 0.5), (0.0, 0.5), (-0.5, 0.5),
                                               (-2.0, 0.0), (-1.0, -1.0), (0.0, -2.0), (1.0, -1.0)])

            border_pixels = np.arange(16)
            border = grids.GridBorder(border_pixels, polynomial_degree=3)

            relocated_coordinates = border.relocate_coordinates_outside_border(coordinates)

            assert relocated_coordinates[0:20] == pytest.approx(coordinates[0:20], 1e-3)
            assert relocated_coordinates[20] == pytest.approx(np.array([-1.0, 0.0]), 1e-3)
            assert relocated_coordinates[21] == pytest.approx(
                np.array([-0.5 * np.sqrt(2), -0.5 * np.sqrt(2)]), 1e-3)
            assert relocated_coordinates[22] == pytest.approx(np.array([0.0, -1.0]), 1e-3)
            assert relocated_coordinates[23] == pytest.approx(
                np.array([0.5 * np.sqrt(2), -0.5 * np.sqrt(2)]), 1e-3)

        def test__change_pixel_order_and_border_pixels__works_as_above(self):
            thetas = np.linspace(0.0, 2.0 * np.pi, 16)
            circle = list(map(lambda x: (np.cos(x), np.sin(x)), thetas))
            coordinates = np.asarray([(-2.0, 0.0), (-1.0, -1.0), (0.0, -2.0), (1.0, -1.0)] + circle + \
                                     [(0.5, 0.0), (0.5, 0.5), (0.0, 0.5), (-0.5, 0.5)])

            border_pixels = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
            border = grids.GridBorder(border_pixels, polynomial_degree=3)

            relocated_coordinates = border.relocate_coordinates_outside_border(coordinates)

            assert relocated_coordinates[0] == pytest.approx(np.array([-1.0, 0.0]), 1e-3)
            assert relocated_coordinates[1] == pytest.approx(
                np.array([-0.5 * np.sqrt(2), -0.5 * np.sqrt(2)]), 1e-3)
            assert relocated_coordinates[2] == pytest.approx(np.array([0.0, -1.0]), 1e-3)
            assert relocated_coordinates[3] == pytest.approx(np.array([0.5 * np.sqrt(2), -0.5 * np.sqrt(2)]),
                                                             1e-3)
            assert relocated_coordinates[4:24] == pytest.approx(coordinates[4:24], 1e-3)

        def test__sub_pixels_in_border__are_not_relocated(self):
            coordinates = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])

            sub_coordinates = np.array([[[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]],
                                        [[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]],
                                        [[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]],
                                        [[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]]])

            border_pixels = np.array([0, 1, 2, 3])
            border = grids.GridBorder(border_pixels, polynomial_degree=3)

            relocated_coordinates = border.relocate_coordinates_outside_border(coordinates)
            relocated_sub_coordinates = border.relocate_sub_coordinates_outside_border(coordinates, sub_coordinates)

            assert relocated_coordinates == pytest.approx(coordinates, 1e-3)
            assert relocated_sub_coordinates == pytest.approx(sub_coordinates, 1e-3)

        def test__sub_pixels_outside_border__are_relocated(self):
            coordinates = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])

            sub_coordinates = np.array([[[2.0, 0.0], [0.2, 0.0], [2.0, 2.0], [0.4, 0.0]],
                                        [[0.0, 2.0], [-2.0, 2.0], [0.3, 0.0], [0.4, 0.0]],
                                        [[-2.0, 0.0], [0.2, 0.0], [0.3, 0.0], [2.0, -2.0]],
                                        [[0.0, -2.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]]])

            border_pixels = np.array([0, 1, 2, 3])
            border = grids.GridBorder(border_pixels, polynomial_degree=3)

            relocated_coordinates = border.relocate_coordinates_outside_border(coordinates)
            relocated_sub_coordinates = border.relocate_sub_coordinates_outside_border(coordinates, sub_coordinates)

            assert relocated_coordinates == pytest.approx(coordinates, 1e-3)

            assert (relocated_sub_coordinates[0, 0] == pytest.approx(np.array([1.0, 0.0]), 1e-3))
            assert (relocated_sub_coordinates[0, 1] == sub_coordinates[0, 1]).all()
            assert (relocated_sub_coordinates[0, 2] == pytest.approx(np.array([0.707, 0.707]), 1e-3))
            assert (relocated_sub_coordinates[0, 3] == sub_coordinates[0, 3]).all()

            assert (relocated_sub_coordinates[1, 0] == pytest.approx(np.array([0.0, 1.0]), 1e-3))
            assert (relocated_sub_coordinates[1, 1] == pytest.approx(np.array([-0.707, 0.707]), 1e-3))
            assert (relocated_sub_coordinates[1, 2] == sub_coordinates[1, 2]).all()
            assert (relocated_sub_coordinates[1, 3] == sub_coordinates[1, 3]).all()

            assert (relocated_sub_coordinates[2, 0] == pytest.approx(np.array([-1.0, 0.0]), 1e-3))
            assert (relocated_sub_coordinates[2, 1] == sub_coordinates[2, 1]).all()
            assert (relocated_sub_coordinates[2, 2] == sub_coordinates[2, 2]).all()
            assert (relocated_sub_coordinates[2, 3] == pytest.approx(np.array([0.707, -0.707]), 1e-3))

            assert (relocated_sub_coordinates[3, 0] == pytest.approx(np.array([0.0, -1.0]), 1e-3))
            assert (relocated_sub_coordinates[3, 1] == sub_coordinates[3, 1]).all()
            assert (relocated_sub_coordinates[3, 2] == sub_coordinates[3, 2]).all()
            assert (relocated_sub_coordinates[3, 3] == sub_coordinates[3, 3]).all()
