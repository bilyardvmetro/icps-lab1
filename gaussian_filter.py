"""Gaussian smoothing: explicit Python convolution and OpenCV reference."""

import math
from dataclasses import dataclass
from numbers import Integral

import cv2
import numpy as np


@dataclass(frozen=True)
class GaussianParams:
    """Odd kernel dimensions; sigma_y defaults to sigma_x."""

    width: int = 5
    height: int = 5
    sigma_x: float = 1.0
    sigma_y: float = None

    def __post_init__(self):
        for size in (self.width, self.height):
            if isinstance(size, bool) or not isinstance(size, Integral) or size < 1 or size % 2 == 0:
                raise ValueError("Размеры ядра должны быть положительными нечётными целыми числами")
        if self.sigma_y is None:
            object.__setattr__(self, "sigma_y", self.sigma_x)
        for sigma in (self.sigma_x, self.sigma_y):
            if not math.isfinite(sigma) or sigma <= 0:
                raise ValueError("Sigma должна быть конечным числом больше нуля")


def gaussian_kernel_1d(size, sigma):
    """Build and normalize weights with Python math, without library filters."""
    GaussianParams(size, 1, sigma, sigma)
    radius = size // 2
    distances = [(i - radius) / sigma for i in range(size)]
    weights = [math.exp(-0.5 * distance * distance) for distance in distances]
    total = sum(weights)
    return [weight / total for weight in weights]


def gaussian_kernel(params):
    """Return the 2D kernel as nested Python lists (rows, columns)."""
    kx = gaussian_kernel_1d(params.width, params.sigma_x)
    ky = gaussian_kernel_1d(params.height, params.sigma_y)
    return [[y * x for x in kx] for y in ky]


def _validate_image(image):
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("Ожидается массив H×W или H×W×C")
    if image.size == 0 or (image.ndim == 3 and image.shape[2] not in (1, 3)):
        raise ValueError("Изображение должно быть непустым, с 1 или 3 каналами")
    if image.dtype not in (np.dtype('uint8'), np.dtype('float64')):
        raise ValueError("Поддерживаются uint8 и float64")
    if not np.isfinite(image).all():
        raise ValueError("Изображение содержит NaN или бесконечность")


def gaussian_native(image, params):
    """Two separable passes using only scalar Python arithmetic in the filter.

    NumPy is used only for input validation and conversion. All weights,
    border indices and weighted sums are calculated explicitly in Python.
    Borders replicate the nearest pixel. The result is always float64.
    """
    _validate_image(image)
    source = image.astype(np.float64, copy=False)
    gray = source.ndim == 2
    if gray:
        source = source[:, :, None]
    height, width, channels = source.shape
    pixels = source.tolist()
    kx = gaussian_kernel_1d(params.width, params.sigma_x)
    ky = gaussian_kernel_1d(params.height, params.sigma_y)
    # Precompute border indices, including kernels larger than the image.
    xs = [[min(width - 1, max(0, x + i - params.width // 2))
           for i in range(params.width)] for x in range(width)]
    ys = [[min(height - 1, max(0, y + i - params.height // 2))
           for i in range(params.height)] for y in range(height)]
    temp = [[[0.0] * channels for _ in range(width)] for _ in range(height)]
    result = [[[0.0] * channels for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                total = 0.0
                for i, weight in enumerate(kx):
                    total += pixels[y][xs[x][i]][c] * weight
                temp[y][x][c] = total
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                total = 0.0
                for j, weight in enumerate(ky):
                    total += temp[ys[y][j]][x][c] * weight
                result[y][x][c] = total
    output = np.asarray(result, dtype=np.float64)
    return output[:, :, 0] if gray else output


def gaussian_opencv(image, params):
    """Same precision and border convention as the native implementation."""
    _validate_image(image)
    result = cv2.GaussianBlur(
        image.astype(np.float64, copy=False),
        (params.width, params.height),
        sigmaX=params.sigma_x, sigmaY=params.sigma_y,
        borderType=cv2.BORDER_REPLICATE,
    )
    if image.ndim == 3 and image.shape[2] == 1:
        result = result[:, :, None]
    return result


def to_uint8(image):
    """Round only after both passes; avoid integer overflow on conversion."""
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)
