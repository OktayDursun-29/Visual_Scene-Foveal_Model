"""Render the original example using the canonical foveal_scene objects."""

from pathlib import Path

from foveal_scene import Circle, Polygon, Rectangle, VisualScene2D


def main() -> None:
    scene = VisualScene2D(256, 256)
    objects = (
        Circle(center=(50, 50), radius=20, fill="blue"),
        Rectangle(xy=(90, 50, 150, 110), fill="green"),
        Polygon(points=((200, 125), (175, 175), (225, 175)), fill="red"),
    )
    output_path = Path(__file__).resolve().parent.parent / "img" / "visual_scene.png"
    scene.render(objects).save(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
