from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image, ImageFilter


ImageInput = Image.Image | np.ndarray


@dataclass(frozen=True)
class FovealConfig:
    """Parameters for radial foveation."""

    fovea_radius: float = 55.0
    blur_radii: Sequence[float] = (0.0, 10.0)


class BasicFovealModel:
    """Blend an image pyramid so acuity decreases with distance from fixation."""

    def __init__(self, config: FovealConfig | None = None) -> None:
        self.config = config or FovealConfig()
        if len(self.config.blur_radii) != 2 or self.config.blur_radii[0] != 0:
            raise ValueError("blur_radii must contain sharp and peripheral levels")

    def __call__(
        self,
        image: ImageInput,
        fixation: tuple[float, float] | None = None,
        return_numpy: bool | None = None,
    ) -> Image.Image | np.ndarray:
        pil_image = self._to_pil(image)
        if fixation is None:
            fixation = (pil_image.width / 2.0, pil_image.height / 2.0)

        layers = [
            pil_image if radius == 0 else pil_image.filter(ImageFilter.GaussianBlur(radius))
            for radius in self.config.blur_radii
        ]
        result = self._blend_layers(layers, fixation)

        if return_numpy is None:
            return_numpy = isinstance(image, np.ndarray)
        if return_numpy:
            return np.asarray(result)
        return result

    def acuity_mask(
        self,
        size: tuple[int, int],
        fixation: tuple[float, float],
    ) -> np.ndarray:
        """Return 1 in the clear fovea and 0 in the blurred periphery."""

        width, height = size
        xs, ys = np.meshgrid(np.arange(width), np.arange(height))
        distance = np.hypot(xs - fixation[0], ys - fixation[1])
        return (distance < self.config.fovea_radius).astype(np.float32)

    def _blend_layers(
        self,
        layers: Sequence[Image.Image],
        fixation: tuple[float, float],
    ) -> Image.Image:
        width, height = layers[0].size
        xs, ys = np.meshgrid(np.arange(width), np.arange(height))
        distance = np.hypot(xs - fixation[0], ys - fixation[1])

        clear = np.asarray(layers[0])
        peripheral = np.asarray(layers[1])
        mask = (distance < self.config.fovea_radius)[..., None]
        result = np.where(mask, clear, peripheral)
        return Image.fromarray(result.astype(np.uint8), mode=layers[0].mode)

    @staticmethod
    def _to_pil(image: ImageInput) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        array = np.asarray(image)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            return Image.fromarray(array, mode="L").convert("RGB")
        if array.ndim == 3 and array.shape[2] in (3, 4):
            return Image.fromarray(array).convert("RGB")
        raise ValueError("image must be a PIL image or a 2D/3D NumPy image array")
