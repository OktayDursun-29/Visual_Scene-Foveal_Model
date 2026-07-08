import numpy as np
from PIL import Image, ImageDraw
import csv
from ReceptiveFields import generate_receptive_fields, extract_rf_statistics

# Defining the scene
class Circle:
    def __init__(self, x, y, radius, color):
        self.x = float(x)
        self.y = float(y)
        self.radius = float(radius)
        self.color = color  # Expected as a tuple: (R, G, B)

class Square:
    def __init__(self, x, y, size, color):
        self.x = float(x)
        self.y = float(y)
        self.size = float(size)
        self.color = color  # Expected as a tuple: (R, G, B)

class Triangle:
    def __init__(self, x, y, size, color):
        self.x = float(x)
        self.y = float(y)
        self.size = float(size)
        self.color = color  # Expected as a tuple: (R, G, B)

class Scene:
    def __init__(self, background_color=(255, 255, 255)):
        self.background_color = background_color
        self.objects = []

    def add_object(self, shape):
        self.objects.append(shape)

    def render_to_image(self, width, height):
        """
        Acts as the camera: takes the mathematical shapes and 
        renders them into a grid of raw RGB pixels.
        """
        # Create a blank image using Pillow
        img = Image.new("RGB", (width, height), self.background_color)
        draw = ImageDraw.Draw(img)

        # Draw every object in the scene
        for obj in self.objects:
            if isinstance(obj, Circle):
                # Pillow requires a bounding box to draw a circle: [x_min, y_min, x_max, y_max]
                bbox = [
                    obj.x - obj.radius, 
                    obj.y - obj.radius, 
                    obj.x + obj.radius, 
                    obj.y + obj.radius
                ]
                draw.ellipse(bbox, fill=obj.color)
                
        return img
