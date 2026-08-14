from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from PIL import Image, ImageDraw

from .objects import Color, DetailLevel, SceneObject


@dataclass(frozen=True)
class VisualScene2D:
    """A simple 2D visual scene that renders shapes onto an image."""

    width: int
    height: int
    background: Color = (255, 255, 255)
    mode: str = "RGB"

    def blank(self) -> Image.Image:
        return Image.new(self.mode, (self.width, self.height), self.background)

    def render(
        self,
        objects: Iterable[SceneObject],
        detail: DetailLevel = "high",
        base_image: Image.Image | None = None,
    ) -> Image.Image:
        image = base_image.copy() if base_image is not None else self.blank()
        if image.size != (self.width, self.height):
            raise ValueError(
                f"base_image size {image.size} does not match scene size "
                f"{(self.width, self.height)}"
            )

        draw = ImageDraw.Draw(image)
        for obj in objects:
            obj.draw(draw, detail=detail)
        return image
