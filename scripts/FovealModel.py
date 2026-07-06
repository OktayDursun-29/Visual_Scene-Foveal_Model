from PIL import Image, ImageDraw, ImageFilter
import numpy as np


class FovealModel:

    def __init__(self, fovea_radius=40, peripheral_radius=90):
        self.fovea_radius = fovea_radius
        self.peripheral_radius = peripheral_radius

    def apply(self, image, fixation):
        # creating blurred versions of the image for different levels of detail
        blur_medium = image.filter(
            ImageFilter.GaussianBlur(radius=4)
        )
        blur_strong = image.filter(
            ImageFilter.GaussianBlur(radius=10)
        )

        original = np.array(image)
        medium = np.array(blur_medium)
        strong = np.array(blur_strong)

        # Create a grid of distances from the fixation point for the blurring
        height, width = original.shape[:2]
        y, x = np.ogrid[:height, :width]
        fx, fy = fixation

        distance = np.sqrt(
            (x - fx) ** 2 +
            (y - fy) ** 2
        )

        # Establish masks based on distance from fixation
        periphery = distance >= self.peripheral_radius
        middle = (
            (distance >= self.fovea_radius) & 
            (distance < self.peripheral_radius)
        )

        # 0–40 pixels: sharp/no blurring, 40–90 pixels: medium/some blurring, 90+ pixels: strong/intense blurring
        result = original.copy()
        result[middle] = medium[middle]
        result[periphery] = strong[periphery]

        return Image.fromarray(result)

