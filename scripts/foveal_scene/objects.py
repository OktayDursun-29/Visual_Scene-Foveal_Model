from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Protocol, Sequence

from PIL import ImageDraw

Color = tuple[int, int, int] | tuple[int, int, int, int] | str
DetailLevel = Literal["low", "high"]
Point = tuple[float, float]


class SceneObject(Protocol):
    """Drawable 2D object representation."""

    def draw(self, draw: ImageDraw.ImageDraw, detail: DetailLevel = "high") -> None:
        """Draw this object onto a PIL image drawing context."""


class MovableSceneObject(SceneObject, Protocol):
    """A drawable object with a position and constant velocity."""

    velocity: Point

    def translated(self, displacement: Point) -> "MovableSceneObject":
        """Return a copy moved by ``displacement``."""


@dataclass(frozen=True)
class Circle:
    center: Point
    radius: float
    fill: Color
    outline: Color | None = None
    width: int = 2
    velocity: Point = (0.0, 0.0)

    def translated(self, displacement: Point) -> "Circle":
        dx, dy = displacement
        x, y = self.center
        return replace(self, center=(x + dx, y + dy))

    def draw(self, draw: ImageDraw.ImageDraw, detail: DetailLevel = "high") -> None:
        x, y = self.center
        box = (x - self.radius, y - self.radius, x + self.radius, y + self.radius)
        line_width = self.width if detail == "high" else max(1, self.width // 2)
        outline = self.outline if detail == "high" else None
        draw.ellipse(box, fill=self.fill, outline=outline, width=line_width)
        if detail == "high" and self.outline:
            inner_radius = self.radius * 0.45
            inner = (
                x - inner_radius,
                y - inner_radius,
                x + inner_radius,
                y + inner_radius,
            )
            draw.ellipse(inner, outline=self.outline, width=max(1, self.width))


@dataclass(frozen=True)
class Rectangle:
    xy: tuple[float, float, float, float]
    fill: Color
    outline: Color | None = None
    width: int = 2
    radius: int = 0
    velocity: Point = (0.0, 0.0)

    def translated(self, displacement: Point) -> "Rectangle":
        dx, dy = displacement
        x0, y0, x1, y1 = self.xy
        return replace(self, xy=(x0 + dx, y0 + dy, x1 + dx, y1 + dy))

    def draw(self, draw: ImageDraw.ImageDraw, detail: DetailLevel = "high") -> None:
        line_width = self.width if detail == "high" else max(1, self.width // 2)
        outline = self.outline if detail == "high" else None
        if self.radius:
            draw.rounded_rectangle(
                self.xy,
                radius=self.radius,
                fill=self.fill,
                outline=outline,
                width=line_width,
            )
        else:
            draw.rectangle(self.xy, fill=self.fill, outline=outline, width=line_width)
        if detail == "high" and self.outline:
            x0, y0, x1, y1 = self.xy
            draw.line((x0, y0, x1, y1), fill=self.outline, width=1)
            draw.line((x0, y1, x1, y0), fill=self.outline, width=1)


@dataclass(frozen=True)
class Polygon:
    points: Sequence[Point]
    fill: Color
    outline: Color | None = None
    width: int = 2
    velocity: Point = (0.0, 0.0)

    def translated(self, displacement: Point) -> "Polygon":
        dx, dy = displacement
        return replace(self, points=tuple((x + dx, y + dy) for x, y in self.points))

    def draw(self, draw: ImageDraw.ImageDraw, detail: DetailLevel = "high") -> None:
        draw.polygon(self.points, fill=self.fill)
        if detail == "high" and self.outline:
            closed = [*self.points, self.points[0]]
            draw.line(closed, fill=self.outline, width=self.width, joint="curve")


@dataclass(frozen=True)
class Line:
    points: Sequence[Point]
    fill: Color
    width: int = 3
    velocity: Point = (0.0, 0.0)

    def translated(self, displacement: Point) -> "Line":
        dx, dy = displacement
        return replace(self, points=tuple((x + dx, y + dy) for x, y in self.points))

    def draw(self, draw: ImageDraw.ImageDraw, detail: DetailLevel = "high") -> None:
        line_width = self.width if detail == "high" else max(1, self.width // 2)
        draw.line(self.points, fill=self.fill, width=line_width, joint="curve")
