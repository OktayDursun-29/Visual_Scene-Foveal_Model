from PIL import Image, ImageDraw, ImageFilter
import numpy as np


class Object2D:
    def __init__(self, shape, x, y, size):
        self.shape = shape
        self.x = x
        self.y = y
        self.size = size

scene = [
    Object2D("circle", 50, 50, 20),
    Object2D("square", 120, 80, 30),
    Object2D("triangle", 200, 150, 25)
]

img = Image.new("RGB", (256, 256), "white")
draw = ImageDraw.Draw(img)

for obj in scene:
    if obj.shape == "circle":
        # Draw an circle using the bounding box defined by the size
        draw.ellipse(
            (obj.x - obj.size, obj.y - obj.size, obj.x + obj.size, obj.y + obj.size),
            outline="black", 
            fill="blue"
        )
    elif obj.shape == "square":
        # Draw a square using the bounding box defined by the size
        draw.rectangle(
            (obj.x - obj.size, obj.y - obj.size, obj.x + obj.size, obj.y + obj.size),
            outline="black", 
            fill="green"
        )
    elif obj.shape == "triangle":
        # Polygons take a list of points, listed the three corners of the triangle
        draw.polygon(
            [
                (obj.x, obj.y - obj.size),
                (obj.x - obj.size, obj.y + obj.size),
                (obj.x + obj.size, obj.y + obj.size)
            ],
            outline="black", 
            fill="red"
        )

img.show()

img.save("img/visual_scene.png")
print("Saved img/visual_scene.png")


