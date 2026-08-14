import unittest

import numpy as np
from PIL import Image

from foveal_scene import BasicFovealModel, Circle, FoveatedObjectRenderer, Rectangle, VisualScene2D


class FovealSceneTests(unittest.TestCase):
    def test_scene_renders_objects_to_image(self):
        scene = VisualScene2D(width=120, height=80, background=(255, 255, 255))
        image = scene.render([Circle(center=(40, 40), radius=15, fill=(255, 0, 0))])

        self.assertEqual(image.size, (120, 80))
        self.assertNotEqual(image.getpixel((40, 40)), (255, 255, 255))

    def test_foveal_model_accepts_numpy_and_preserves_shape(self):
        model = BasicFovealModel()
        image = np.zeros((64, 96, 3), dtype=np.uint8)
        image[:, 48:] = 255

        result = model(image, fixation=(48, 32))

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, image.shape)

    def test_object_renderer_keeps_foveal_detail(self):
        scene = VisualScene2D(width=160, height=120, background=(245, 245, 245))
        objects = [
            Rectangle(
                xy=(20, 20, 90, 90),
                fill=(80, 140, 220),
                outline=(0, 0, 0),
                width=4,
            )
        ]
        renderer = FoveatedObjectRenderer(scene)

        high = scene.render(objects, detail="high")
        foveated = renderer.render(objects, fixation=(25, 25))

        self.assertEqual(foveated.size, high.size)
        self.assertEqual(foveated.getpixel((20, 20)), high.getpixel((20, 20)))

    def test_pil_input_returns_pil_image(self):
        model = BasicFovealModel()
        image = Image.new("RGB", (50, 50), (120, 120, 120))

        result = model(image)

        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, image.size)


if __name__ == "__main__":
    unittest.main()
