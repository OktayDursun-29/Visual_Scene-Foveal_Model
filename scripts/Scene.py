import numpy as np
from PIL import Image, ImageDraw

# ==========================
# Shape Definitions
# ==========================
class Circle:
    def __init__(self, x, y, radius, color):
        self.x, self.y, self.radius, self.color = float(x), float(y), float(radius), color

    def contains(self, x, y):
        """Checks whether a point lies inside the circle."""
        return (x - self.x)**2 + (y - self.y)**2 <= self.radius**2


class Square:
    def __init__(self, x, y, size, color):
        self.x, self.y, self.size, self.color = float(x), float(y), float(size), color

    def contains(self, x, y):
        """Checks whether a point lies inside the square."""
        half = self.size / 2
        return (self.x - half <= x <= self.x + half) and (self.y - half <= y <= self.y + half)


class Triangle:
    def __init__(self, x, y, size, color):
        self.x, self.y, self.size, self.color = float(x), float(y), float(size), color

    def get_vertices(self):
        """Creates the three vertices of an upright triangle."""
        half = self.size / 2
        return [(self.x, self.y - half), (self.x - half, self.y + half), (self.x + half, self.y + half)]

    def contains(self, x, y):
        """Uses barycentric coordinates to determine if a point lies inside the triangle."""
        (x1, y1), (x2, y2), (x3, y3) = self.get_vertices()
        denom = ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
        a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / denom
        b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / denom
        return (0 <= a <= 1) and (0 <= b <= 1) and (0 <= (1 - a - b) <= 1)


# ==========================
# Scene Definition
# ==========================
class Scene:
    def __init__(self, background_color=(255,255,255)):
        self.background_color = background_color
        self.objects = []

    def add_object(self, shape):
        self.objects.append(shape)

    def render_to_image(self, width, height):
        """Converts the mathematical scene into an RGB image."""
        img = Image.new("RGB", (width, height), self.background_color)
        draw = ImageDraw.Draw(img)

        # Draw objects in order added. Later objects overwrite earlier ones.
        for obj in self.objects:
            if isinstance(obj, Circle):
                draw.ellipse([obj.x - obj.radius, obj.y - obj.radius, obj.x + obj.radius, obj.y + obj.radius], fill=obj.color)
            elif isinstance(obj, Square):
                half = obj.size / 2
                draw.rectangle([obj.x - half, obj.y - half, obj.x + half, obj.y + half], fill=obj.color)
            elif isinstance(obj, Triangle):
                draw.polygon(obj.get_vertices(), fill=obj.color)
                
        return img