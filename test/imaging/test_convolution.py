import numpy as np
from src.imaging import convolution
from src.imaging import mask, scaled_array
import pytest
from src import exc


@pytest.fixture(name="simple_number_array")
def make_simple_number_array():
    return np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])


@pytest.fixture(name="cross_mask")
def make_cross_mask():
    mask = np.full((3, 3), False)

    mask[0, 0] = True
    mask[0, 2] = True
    mask[2, 2] = True
    mask[2, 0] = True

    return mask


@pytest.fixture(name="cross_number_array")
def make_cross_number_array():
    return np.array([[-1, 0, -1], [1, 2, 3], [-1, 4, -1]])


@pytest.fixture(name="simple_frame_array")
def make_simple_frame_array(simple_frame_maker):
    return simple_frame_maker.make_frame_array((3, 3))


@pytest.fixture(name="cross_frame_array")
def make_cross_frame_array(cross_frame_maker):
    return cross_frame_maker.make_frame_array((3, 3))


@pytest.fixture(name="cross_mask_frame_array")
def make_cross_mask_frame_array(cross_frame_maker):
    return cross_frame_maker.make_blurring_frame_array((3, 3), blurring_region_mask=np.full((3, 3), False))


@pytest.fixture(name="simple_frame_maker")
def make_simple_frame_maker():
    return convolution.FrameMaker(np.full((3, 3), False))


@pytest.fixture(name="cross_frame_maker")
def make_cross_frame_maker(cross_mask):
    return convolution.FrameMaker(cross_mask)


@pytest.fixture(name="simple_kernel")
def make_simple_kernel():
    return np.array([[0, 0.1, 0], [0.1, 0.6, 0.1], [0, 0.1, 0]])


class TestNumbering(object):
    def test_simple_numbering(self, simple_number_array):
        shape = (3, 3)

        frame_maker = convolution.FrameMaker(np.full(shape, False))

        number_array = frame_maker.number_array

        assert number_array.shape == shape
        # noinspection PyUnresolvedReferences
        assert (number_array == simple_number_array).all()

    def test_simple_mask(self, cross_mask):
        frame_maker = convolution.FrameMaker(cross_mask)

        assert (frame_maker.number_array == np.array([[-1, 0, -1], [1, 2, 3], [-1, 4, -1]])).all()

    def test_even_failure(self):
        with pytest.raises(exc.KernelException):
            convolution.FrameMaker(np.full((3, 3), False)).convolver_for_kernel_shape((2, 2), None)

    def test_mismatching_masks_failure(self, cross_frame_maker):
        with pytest.raises(AssertionError):
            cross_frame_maker.make_blurring_frame_array((3, 3), np.full((3, 4), False))


class TestFrameExtraction(object):
    def test_trivial_frame_at_coords(self, simple_frame_maker):
        assert (np.array([i for i in range(9)]) == simple_frame_maker.frame_at_coords(coords=(1, 1),
                                                                                      kernel_shape=(3, 3))).all()

    def test_corner_frame(self, simple_frame_maker):
        corner_frame = [-1, -1, -1,
                        -1, 0, 1,
                        -1, 3, 4]

        result = simple_frame_maker.frame_at_coords(coords=(0, 0), kernel_shape=(3, 3))

        assert (corner_frame == result).all()

    def test_simple_square(self, simple_frame_maker):
        frame_array = simple_frame_maker.make_frame_array(kernel_shape=(3, 3))

        assert 9 == len(frame_array)

        assert (np.array([i for i in range(9)] == frame_array[4])).all()

    def test_masked_square(self, cross_frame_maker):
        frame_array = cross_frame_maker.make_frame_array(kernel_shape=(3, 3))

        assert 5 == len(frame_array)

        assert (np.array([-1, -1, -1,
                          -1, 0, -1,
                          1, 2, 3]) == frame_array[0]).all()

        assert (np.array([-1, -1, 0,
                          -1, 1, 2,
                          -1, -1, 4]) == frame_array[1]).all()

        assert (np.array([0, -1, -1,
                          2, 3, -1,
                          4, -1, -1]) == frame_array[3]).all()

        assert (np.array([1, 2, 3,
                          -1, 4, -1,
                          -1, -1, -1]) == frame_array[4]).all()

    def test_masked_square_masked_frame_array(self, cross_frame_maker):
        masked_frame_array = cross_frame_maker.make_blurring_frame_array(kernel_shape=(3, 3),
                                                                         blurring_region_mask=np.full((3, 3), False))

        assert 4 == len(masked_frame_array)

        assert (np.array([-1, -1, -1,
                          -1, -1, 0,
                          -1, 1, 2]) == masked_frame_array[0]).all()
        assert (np.array([2, 3, -1,
                          4, -1, -1,
                          -1, -1, -1]) == masked_frame_array[-1]).all()


class TestBlurringRegionMask(object):
    def test_no_blurring_region(self, cross_mask):
        frame_maker = convolution.FrameMaker(cross_mask)

        # noinspection PyTypeChecker
        assert (len(frame_maker.make_blurring_frame_array(kernel_shape=(3, 3), blurring_region_mask=cross_mask)) == 0)

    def test_partial_blurring_region(self, cross_mask):
        partial_mask = np.array(cross_mask)
        partial_mask[0, 0] = False

        frame_maker = convolution.FrameMaker(cross_mask)
        masked_frame_array = frame_maker.make_blurring_frame_array(kernel_shape=(3, 3),
                                                                   blurring_region_mask=partial_mask)

        assert (np.array([-1, -1, -1,
                          -1, -1, 0,
                          -1, 1, 2]) == masked_frame_array[0]).all()

    def test_no_blurring_region_mask(self, cross_frame_maker):
        frame_array = cross_frame_maker.make_blurring_frame_array(kernel_shape=(3, 3),
                                                                  blurring_region_mask=np.full((3, 3), False))
        assert len(frame_array) == 4


class TestConvolution(object):
    def test_shortcut(self):
        msk = mask.Mask.circular((10, 10), 1, 2)
        psf = scaled_array.ScaledArray.single_value(0.1, (3, 3))
        frame = convolution.FrameMaker(mask=msk)
        kernel_convolver_shortcut = frame.convolver_for_kernel(psf)
        kernel_convolver = frame.convolver_for_kernel_shape(kernel_shape=psf.shape,
                                                            blurring_region_mask=msk.blurring_mask_for_kernel_shape(
                                                                kernel_shape=psf.shape)).convolver_for_kernel(
            kernel=psf)

        assert len(kernel_convolver.frame_array) == len(kernel_convolver.frame_array)
        assert (kernel_convolver.frame_array[0] == kernel_convolver_shortcut.frame_array[0]).all()

    def test_simple_convolution(self, simple_frame_array, simple_kernel):
        convolver = convolution.Convolver(simple_frame_array, [])

        result = convolver.convolver_for_kernel(
            simple_kernel).convolution_for_value_frame_and_new_array(1, convolver.frame_array[4], np.zeros((9,)))

        assert (result == np.array([0, 0.1, 0,
                                    0.1, 0.6, 0.1,
                                    0, 0.1, 0])).all()

    def test_full_convolution(self, simple_frame_array):
        pixel_array = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])

        kernel = np.array([[0, 0, 0], [0, 0.5, 0.5], [0, 0, 0]])

        convolver = convolution.Convolver(simple_frame_array, [])

        result = convolver.convolver_for_kernel(kernel).convolve_array(pixel_array, [])

        assert (result == np.array([0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5])).all()

    def test_cross_mask_convolution(self, cross_frame_array):
        pixel_array = np.array([0,
                                0, 1, 0,
                                0])

        kernel = np.array([[0, 0, 0],
                           [0, 0.5, 0.5],
                           [0, 0, 0]])

        convolver = convolution.Convolver(cross_frame_array, [])

        result = convolver.convolver_for_kernel(kernel).convolve_array(pixel_array, [])

        assert (result == np.array([0,
                                    0, 0.5, 0.5,
                                    0])).all()


@pytest.fixture(name="convolver_4_simple")
def make_convolver_4_simple():
    shape = (4, 4)
    mask = np.full(shape, False)

    frame_maker = convolution.FrameMaker(mask)
    return frame_maker.convolver_for_kernel_shape((3, 3), mask)


@pytest.fixture(name="convolver_4_edges")
def make_convolver_4_edges():
    mask = np.array(
        [[True, True, True, True],
         [True, False, False, True],
         [True, False, False, True],
         [True, True, True, True]]
    )

    frame_maker = convolution.FrameMaker(mask)
    return frame_maker.convolver_for_kernel_shape((3, 3), mask)


class TestNonTrivialExamples(object):
    def test_larger_mask(self, convolver_4_simple):
        kernel = np.array([[0, 0.2, 0],
                           [0.2, 0.4, 0.2],
                           [0, 0.2, 0]])

        pixel_array = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

        kernel_convolver = convolver_4_simple.convolver_for_kernel(kernel)

        result = kernel_convolver.convolve_array(pixel_array, [])

        assert (result == np.array([0, 0, 0, 0, 0, 0.2, 0, 0, 0.2, 0.4, 0.2, 0, 0, 0.2, 0, 0])).all()

    def test_asymmetric_kernel(self, convolver_4_simple):

        asymmetric_kernel = np.array([[0, 0.0, 0],
                                      [0.4, 0.2, 0.3],
                                      [0, 0.1, 0]])

        pixel_array = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

        kernel_convolver = convolver_4_simple.convolver_for_kernel(asymmetric_kernel)

        result = kernel_convolver.convolve_array(pixel_array, [])

        assert (result == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0.4, 0.2, 0.3, 0, 0, 0.1, 0, 0])).all()

    def test_two_pixel_sum(self, convolver_4_simple):
        kernel = np.array([[0, 0.2, 0],
                           [0.2, 0.4, 0.2],
                           [0, 0.2, 0]])

        pixel_array = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0])

        kernel_convolver = convolver_4_simple.convolver_for_kernel(kernel)

        result = kernel_convolver.convolve_array(pixel_array, blurring_array=[])

        assert (result == np.array([0, 0, 0.2, 0, 0, 0.4, 0.4, 0.2, 0.2, 0.4, 0.4, 0, 0, 0.2, 0, 0])).all()

    def test_two_pixel_sum_masked(self, convolver_4_edges):
        kernel = np.array([[0, 0.2, 0],
                           [0.2, 0.4, 0.2],
                           [0, 0.2, 0]])

        pixel_array = np.array([
            1, 1,
            0, 0
        ])

        kernel_convolver = convolver_4_edges.convolver_for_kernel(kernel)

        result = kernel_convolver.convolve_array(pixel_array, blurring_array=[])

        assert (np.round(result, 1) == np.array([
            0.6, 0.6,
            0.2, 0.2
        ])).all()

    def test_cross_mask_with_blurring_entries(self, cross_frame_maker):
        kernel = np.array([[0, 0.2, 0],
                           [0.2, 0.4, 0.2],
                           [0, 0.2, 0]])
        pixel_array = np.array([1, 0, 0, 0, 0])

        blurring_array = np.array([1, 0, 0, 0])

        convolver = cross_frame_maker.convolver_for_kernel_shape((3, 3), np.full((3, 3), False))
        kernel_convolver = convolver.convolver_for_kernel(kernel)

        result = kernel_convolver.convolve_array(pixel_array, blurring_array)

        assert (np.round(result, 1) == np.array([0.6, 0.2, 0.2, 0., 0.])).all()


class TestSubConvolution(object):
    def test_calculate_limits(self):
        limits = convolution.calculate_limits((5, 5), (3, 3))
        assert limits == (1, 1, 4, 4)

    def test_is_in_sub_shape(self):
        assert not convolution.is_in_sub_shape(0, (1, 1, 4, 4), (5, 5))
        assert not convolution.is_in_sub_shape(4, (1, 1, 4, 4), (5, 5))
        assert not convolution.is_in_sub_shape(5, (1, 1, 4, 4), (5, 5))
        assert not convolution.is_in_sub_shape(9, (1, 1, 4, 4), (5, 5))
        assert convolution.is_in_sub_shape(6, (1, 1, 4, 4), (5, 5))
        assert convolution.is_in_sub_shape(8, (1, 1, 4, 4), (5, 5))
        assert convolution.is_in_sub_shape(16, (1, 1, 4, 4), (5, 5))
        assert convolution.is_in_sub_shape(18, (1, 1, 4, 4), (5, 5))
        assert not convolution.is_in_sub_shape(21, (1, 1, 4, 4), (5, 5))
        assert not convolution.is_in_sub_shape(24, (1, 1, 4, 4), (5, 5))

    def test_simple_convolution(self):
        convolver = convolution.FrameMaker(mask=np.full((5, 5), False)).convolver_for_kernel_shape(
            (5, 5), blurring_region_mask=np.full((5, 5), False)).convolver_for_kernel(np.ones((5, 5)))

        pixel_array = np.zeros(shape=(25,))

        pixel_array[12] = 1

        convolved_vector = convolver.convolve_array(pixel_array, np.zeros(shape=(0,)), sub_shape=(3, 3))

        assertion_array = np.zeros(shape=(25,))

        assertion_array[6] = 1
        assertion_array[7] = 1
        assertion_array[8] = 1
        assertion_array[11] = 1
        assertion_array[12] = 1
        assertion_array[13] = 1
        assertion_array[16] = 1
        assertion_array[17] = 1
        assertion_array[18] = 1

        assert (assertion_array == convolved_vector).all()


class TestOptionalBlurringRegion(object):
    def test_create_kernel_convolver(self, cross_mask):
        frame_maker = convolution.FrameMaker(cross_mask)
        convolver = frame_maker.convolver_for_kernel_shape((3, 3))
        kernel_convolver = convolver.convolver_for_kernel(np.ones((3, 3)))
        assert kernel_convolver.frame_array is not None
        assert kernel_convolver.blurring_frame_array is None

    def test_convolution(self, cross_mask):
        frame_maker = convolution.FrameMaker(cross_mask)
        convolver = frame_maker.convolver_for_kernel_shape((3, 3))
        kernel = np.array([[0, 0, 0],
                           [0, 0.5, 0.5],
                           [0, 0, 0]])

        pixel_array = np.array([0,
                                0, 1, 0,
                                0])

        kernel_convolver = convolver.convolver_for_kernel(kernel)

        result = kernel_convolver.convolve_array(pixel_array)

        assert (result == np.array([0,
                                    0, 0.5, 0.5,
                                    0])).all()


class TestConvolveMappingMatrix(object):

    def test__asymetric_convolver__matrix_blurred_correctly(self, convolver_4_simple):

        asymmetric_kernel = np.array([[0, 0.0, 0],
                                      [0.4, 0.2, 0.3],
                                      [0, 0.1, 0]])

        mapping = np.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0 ,0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 1, 0], # The 0.3 should be 'chopped' from this pixel as it is on the right-most edge
                            [0, 0, 0],
                            [1, 0, 0],
                            [0, 0, 1],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]])

        kernel_convolver = convolver_4_simple.convolver_for_kernel(asymmetric_kernel)

        blurred_mapping = kernel_convolver.convolve_mapping_matrix(mapping)

        assert (blurred_mapping == np.array([[0,     0,   0],
                                             [0,     0,   0],
                                             [0,     0,   0],
                                             [0,     0,   0],
                                             [0,     0,   0],
                                             [0,     0,   0],
                                             [0,   0.4,   0],
                                             [0,   0.2,   0],
                                             [0.4,   0,   0],
                                             [0.2,   0, 0.4],
                                             [0.3,   0, 0.2],
                                             [0,   0.1, 0.3],
                                             [0,     0,   0],
                                             [0.1,   0,   0],
                                             [0,     0, 0.1],
                                             [0,     0,   0]])).all()

    def test__asymetric_convolver__multiple_overlapping_blurred_entires_in_matrix(self, convolver_4_simple):

        asymmetric_kernel = np.array([[0, 0.0, 0],
                                      [0.4, 0.2, 0.3],
                                      [0, 0.1, 0]])

        mapping = np.array([[0, 1, 0],
                            [0, 1, 0],
                            [0, 1, 0],
                            [0, 0, 0],
                            [0, 0 ,0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 1, 0], # The 0.3 should be 'chopped' from this pixel as it is on the right-most edge
                            [1, 0, 0],
                            [1, 0, 0],
                            [0, 0, 1],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]])

        kernel_convolver = convolver_4_simple.convolver_for_kernel(asymmetric_kernel)

        blurred_mapping = kernel_convolver.convolve_mapping_matrix(mapping)

        assert blurred_mapping == pytest.approx(np.array([[0,     0.6,   0],
                                                          [0,     0.9,   0],
                                                          [0,     0.5,   0],
                                                          [0,     0.3,   0],
                                                          [0,     0.1,   0],
                                                          [0,     0.1,   0],
                                                          [0,   0.5,   0],
                                                          [0,   0.2,   0],
                                                          [0.6,   0,   0],
                                                          [0.5,   0, 0.4],
                                                          [0.3,   0, 0.2],
                                                          [0,   0.1, 0.3],
                                                          [0.1,   0,   0],
                                                          [0.1,   0,   0],
                                                          [0,     0, 0.1],
                                                          [0,     0,   0]]), 1e-4)


class TestJitFunction(object):

    # def test__identical_to_non_jit_version__no_blurring_array(self, cross_frame_maker):
    #
    #     kernel = np.array([[0.6, 0.2, 0],
    #                        [0.2, 0.4, 0.2],
    #                        [2.0, 0.2, 0.1]])
    #     pixel_array = np.array([1, 2, 8, 1, 0])
    #
    #     convolver = cross_frame_maker.convolver_for_kernel_shape((3, 3), np.full((3, 3), False))
    #     kernel_convolver = convolver.convolver_for_kernel(kernel)
    #
    #     result_0 = kernel_convolver.convolve_array(pixel_array)
    #     result_1 = kernel_convolver.convolve_array_jit(pixel_array)
    #
    #     assert (result_0 == result_1).all()

    def test__identical_to_non_jit_version__include_blurring_array(self, cross_frame_maker):

        kernel = np.array([[0.6, 0.2, 0],
                           [0.2, 0.4, 0.2],
                           [2.0, 0.2, 0.1]])
        pixel_array = np.array([1, 2, 8, 1, 0])

        blurring_array = np.array([1, 2, 4, 8])

        convolver = cross_frame_maker.convolver_for_kernel_shape((3, 3), np.full((3, 3), False))
        kernel_convolver = convolver.convolver_for_kernel(kernel)

        result_0 = kernel_convolver.convolve_array(pixel_array, blurring_array)
        result_1 = kernel_convolver.convolve_array_jitted(pixel_array, blurring_array)

        assert result_0 == pytest.approx(result_1, 1e-4)

    def test__identical_to_non_jit_version__mapping_matrix(self, convolver_4_simple):

        asymmetric_kernel = np.array([[1.0, 0.0, 0],
                                      [0.4, 0.2, 0.3],
                                      [0, 0.1, 0]])

        mapping = np.array([[0, 1, 0],
                            [0, 1, 0],
                            [0, 1, 0],
                            [0, 0, 0],
                            [0, 7 ,0],
                            [0, 0, 0],
                            [0, 3, 0],
                            [0, 1, 3], # The 0.3 should be 'chopped' from this pixel as it is on the right-most edge
                            [1, 0, 0],
                            [1, 0, 1],
                            [0, 0, 1],
                            [0, 0, 0],
                            [0, 0, 9],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]])

        kernel_convolver = convolver_4_simple.convolver_for_kernel(asymmetric_kernel)

        result_0 = kernel_convolver.convolve_mapping_matrix(mapping)
        result_1 = kernel_convolver.convolve_mapping_matrix_jit(mapping)

        assert result_0[:,0] == pytest.approx(result_1[:,0], 1e-4)
        assert result_0[:,1] == pytest.approx(result_1[:,1], 1e-4)
        assert result_0[:,2] == pytest.approx(result_1[:,2], 1e-4)