from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from PIL import Image

from .fovea import BasicFovealModel, FovealConfig
from .objects import SceneObject
from .scene import VisualScene2D


@dataclass
class FoveatedObjectRenderer:
    """Render object detail as a function of distance from fixation."""

    scene: VisualScene2D
    foveal_model: BasicFovealModel | None = None

    def __post_init__(self) -> None:
        if self.foveal_model is None:
            self.foveal_model = BasicFovealModel(FovealConfig())

    def render(
        self,
        objects: Iterable[SceneObject],
        fixation: tuple[float, float] | None = None,
    ) -> Image.Image:
        objects = list(objects)
        if fixation is None:
            fixation = (self.scene.width / 2.0, self.scene.height / 2.0)

        high_detail = self.scene.render(objects, detail="high")
        return self.foveal_model(high_detail, fixation=fixation)
