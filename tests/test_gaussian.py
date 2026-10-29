"""Numerical properties, independent direct convolution and OpenCV comparison."""

import unittest

import numpy as np

from gaussian_filter import (
    GaussianParams, gaussian_kernel, gaussian_native, gaussian_opencv, to_uint8,
)


class GaussianTests(unittest.TestCase):
    def test_kernel_normalized_symmetric_positive(self):
        kernel = np.array(gaussian_kernel(GaussianParams(7, 5, 1.8, 0.7)))
        self.assertAlmostEqual(float(kernel.sum()), 1.0, places=14)
        np.testing.assert_allclose(kernel, kernel[::-1, ::-1], atol=1e-15)
        self.assertTrue((kernel > 0).all())

    def test_against_opencv(self):
        rng = np.random.default_rng(42)
        # Degenerate dimensions, grayscale, one channel, RGB, oversized kernels.
        for shape in ((1, 1), (1, 7), (6, 1), (9, 13), (5, 8, 1), (8, 11, 3)):
            for dtype in (np.uint8, np.float64):
                image = rng.integers(0, 256, shape, dtype=np.uint8).astype(dtype)
                for params in (GaussianParams(1, 1), GaussianParams(3, 5, 0.8, 1.7),
                               GaussianParams(15, 9, 3.0, 0.3), GaussianParams(1, 7, 2, 1)):
                    with self.subTest(shape=shape, dtype=dtype, params=params):
                        before = image.copy()
                        actual = gaussian_native(image, params)
                        expected = gaussian_opencv(image, params)
                        self.assertEqual(actual.shape, image.shape)
                        self.assertEqual(actual.dtype, np.float64)
                        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
                        np.testing.assert_array_equal(image, before)

    def test_direct_2d_convolution(self):
        image = np.arange(30, dtype=np.float64).reshape(5, 6)
        params = GaussianParams(5, 3, 1.2, 0.6)
        kernel = gaussian_kernel(params)
        expected = np.zeros_like(image)
        for y in range(5):
            for x in range(6):
                for j in range(3):
                    for i in range(5):
                        expected[y, x] += kernel[j][i] * image[
                            min(4, max(0, y + j - 1)), min(5, max(0, x + i - 2))]
        np.testing.assert_allclose(gaussian_native(image, params), expected, rtol=0, atol=1e-12)

    def test_constant_is_preserved(self):
        image = np.full((7, 8, 3), 127, dtype=np.uint8)
        for function in (gaussian_native, gaussian_opencv):
            np.testing.assert_allclose(function(image, GaussianParams(9, 9, 2)), image, atol=1e-12)

    def test_impulse_and_channel_independence(self):
        image = np.zeros((11, 11, 3), dtype=np.float64)
        image[5, 5, 1] = 1.0
        params = GaussianParams(5, 5, 1)
        actual = gaussian_native(image, params)
        np.testing.assert_allclose(actual[3:8, 3:8, 1], gaussian_kernel(params), atol=1e-15)
        self.assertEqual(float(actual[:, :, 0].sum() + actual[:, :, 2].sum()), 0)
        self.assertAlmostEqual(float(actual.sum()), 1)

    def test_invalid_parameters(self):
        for size in (0, -3, 2, 2.5, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                GaussianParams(size, 3)
        for sigma in (0, -1, float('nan'), float('inf')):
            with self.subTest(sigma=sigma), self.assertRaises(ValueError):
                GaussianParams(sigma_x=sigma)
            with self.assertRaises(ValueError):
                GaussianParams(sigma_y=sigma)

    def test_invalid_images(self):
        for image in (np.zeros((0, 4), dtype=np.uint8), np.zeros(5), np.zeros((2, 2, 4)),
                      np.zeros((3, 3), dtype=np.int32), np.array([[float('nan')]])):
            for function in (gaussian_native, gaussian_opencv):
                with self.subTest(shape=image.shape), self.assertRaises(ValueError):
                    function(image, GaussianParams())

    def test_extreme_positive_sigmas(self):
        for sigma in (1e-300, 1e300):
            kernel = np.asarray(gaussian_kernel(GaussianParams(5, 5, sigma)))
            self.assertTrue(np.isfinite(kernel).all())
            self.assertAlmostEqual(float(kernel.sum()), 1.0)

    def test_round_and_clip(self):
        np.testing.assert_array_equal(to_uint8(np.array([-2, 0.2, 1.8, 256])), [0, 0, 2, 255])


if __name__ == '__main__':
    unittest.main()
