import numpy as np
from shapely.geometry import Point, Polygon


def circle_overlap_area(circle_obj, rf_x, rf_y, rf_radius):
    # Computes the overlap area between a scene circle and a receptive field

    rf = Point(rf_x, rf_y).buffer(rf_radius, resolution=64)
    obj = Point(circle_obj.x, circle_obj.y).buffer(circle_obj.radius, resolution=64)

    return rf.intersection(obj).area


def square_overlap_area(square_obj, rf_x, rf_y, rf_radius):
    # Computes the overlap area between a scene square and a receptive field

    half = square_obj.size / 2

    square = Polygon([
        (square_obj.x - half, square_obj.y - half),
        (square_obj.x + half, square_obj.y - half),
        (square_obj.x + half, square_obj.y + half),
        (square_obj.x - half, square_obj.y + half)
    ])

    rf = Point(rf_x, rf_y).buffer(rf_radius, resolution=64)

    return rf.intersection(square).area

def triangle_overlap_area(triangle_obj, rf_x, rf_y, rf_radius):
    # Computes the overlap area between a scene triangle and a receptive field
    s = triangle_obj.size
    h = np.sqrt(3) * s / 2

    triangle = Polygon([
        (triangle_obj.x, triangle_obj.y - 2*h/3),
        (triangle_obj.x - s/2, triangle_obj.y + h/3),
        (triangle_obj.x + s/2, triangle_obj.y + h/3)
    ])

    rf = Point(rf_x, rf_y).buffer(rf_radius, resolution=64)

    return rf.intersection(triangle).area


def object_overlap_area(obj, rf_x, rf_y, rf_radius):

    if hasattr(obj, "radius"):
        return circle_overlap_area(obj, rf_x, rf_y, rf_radius)

    elif obj.__class__.__name__ == "Square":
        return square_overlap_area(obj, rf_x, rf_y, rf_radius)

    elif obj.__class__.__name__ == "Triangle":
        return triangle_overlap_area(obj, rf_x, rf_y, rf_radius)

    return 0.0


# Generate receptive fields
def generate_receptive_fields(image_shape, fixation, base_radius, growth_rate, overlap_density, target_rf_count=98):
    # Generates receptive fields with a discontinuous size profile.
    # Foveal RFs are much smaller than peripheral RFs.
    # Radius changes in discrete steps instead of continuously.
    height, width = image_shape[:2]
    fix_x, fix_y = fixation

    corners = np.array([[0, 0], [width, 0], [0, height], [width, height]])
    max_dist = np.max(np.linalg.norm(corners - np.array([fix_x, fix_y]), axis=1))

    # Radius step function
    def rf_radius(distance):
        FOVEAL_BOUNDARY = 100
        if distance <= FOVEAL_BOUNDARY:
            return base_radius * 0.7  
        else:
            return base_radius * 1.2  
    
    r0 = rf_radius(0)
    rfs = [{"x": fix_x, "y": fix_y, "r": r0}]
    current_dist = r0 * overlap_density

    while current_dist < max_dist + base_radius:
        r = rf_radius(current_dist)
        circumference = 2 * np.pi * current_dist
        spacing = r * overlap_density
        num_circles = max(1, int(circumference / spacing))
        angles = np.linspace(0, 2 * np.pi, num_circles, endpoint=False)

        for angle in angles:
            cx = fix_x + current_dist * np.cos(angle)
            cy = fix_y + current_dist * np.sin(angle)

            if -r <= cx <= width + r and -r <= cy <= height + r:
                rfs.append({"x": cx, "y": cy, "r": r})
                
                if len(rfs) >= 98:
                    return rfs

        current_dist += spacing

    return rfs


# Predict stats directly from the scene
def predict_rf_stats(scene, rfs):
    # Predict receptive field RGB statistics directly from the mathematical scene
    # using exact area overlap between objects and receptive fields.
    
    statistical_map = []
    background = np.array(scene.background_color, dtype=np.float32)

    for rf in rfs:
        cx = rf["x"]
        cy = rf["y"]
        r = rf["r"]

        rf_area = np.pi * r * r

        weighted_sum = np.zeros(3, dtype=np.float32)
        covered_area = 0.0

        region_colors = []
        region_weights = []

        # Compute contribution from each object
        for obj in scene.objects:

            overlap = object_overlap_area(obj, cx, cy, r)

            if overlap <= 0:
                continue

            color = np.array(obj.color, dtype=np.float32)

            weighted_sum += overlap * color
            covered_area += overlap

            region_colors.append(color)
            region_weights.append(overlap)

        # Add background contribution
        background_area = max(0.0, rf_area - covered_area)

        weighted_sum += background_area * background

        # Mean RGB
        mean_rgb = weighted_sum / rf_area

        # Weighted variance
        variance_rgb = np.zeros(3, dtype=np.float32)

        for color, area in zip(region_colors, region_weights):
            weight = area / rf_area
            variance_rgb += weight * (color - mean_rgb) ** 2

        background_weight = background_area / rf_area
        variance_rgb += background_weight * (background - mean_rgb) ** 2

        variance_rgb = np.maximum(variance_rgb, 1e-6)

        statistical_map.append({
            "rf": rf,
            "rf_area": rf_area,
            "mean": mean_rgb,
            "variance": variance_rgb
        })

    return statistical_map