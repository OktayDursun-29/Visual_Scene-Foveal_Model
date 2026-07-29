#!/usr/bin/env python3

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.random import normal


def circle_circle_intersection(x1, y1, r1, x2, y2, r2):
    # Area of intersection between two circles.
    # Returns the area of the portion of the FIRST circle overlapped
    # by the second circle.
    
    d = jnp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Case 1 : no overlap 
    no_overlap = d >= (r1 + r2)

    # Case 2 : one inside the other 
    contained = d <= jnp.abs(r1 - r2)

    contained_area = jnp.where(
        r1 <= r2,
        jnp.pi * r1 ** 2,      # first circle completely inside second
        jnp.pi * r2 ** 2       # second completely inside first
    )

    # General Case
    eps = 1e-8

    alpha = jnp.arccos(jnp.clip((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1 + eps), -1.0, 1.0))

    beta = jnp.arccos(jnp.clip((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2 + eps), -1.0, 1.0))

    term = ((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))

    lens_area = (r1 ** 2 * alpha + r2 ** 2 * beta - 0.5 * jnp.sqrt(jnp.maximum(term, 0.0)))

    return jnp.where(no_overlap, 0.0,
         jnp.where(
            contained,
            contained_area,
            lens_area,
        ),
    )


@jit
def predict_rf_stats(rfs, objects, background_color):
    '''
    MAIN FUNCTION
     Args:
        rfs: (N_RF,3) -> [x, y, r]
    objects: (N_OBJ,7) -> [x, y, size, type, r, g, b]
    '''
    obj_x = objects[:, 0]
    obj_y = objects[:, 1]
    obj_size = objects[:, 2]
    obj_type = objects[:, 3]
    obj_colors = objects[:, 4:7]

    # Equivalent radii

    circle_radius = obj_size

    square_radius = jnp.sqrt((obj_size ** 2) / jnp.pi)

    triangle_radius = jnp.sqrt(((jnp.sqrt(3.0) / 4.0) * obj_size ** 2) / jnp.pi)

    eq_radii = jnp.where(obj_type == 0, circle_radius,
        jnp.where(
            obj_type == 1,
            square_radius,
            triangle_radius,
        ),
    )

    def compute_single_rf(rf_row):

        rf_x, rf_y, rf_r = rf_row
        rf_area = jnp.pi * rf_r ** 2

        overlaps = vmap(circle_circle_intersection)(jnp.full_like(obj_x, rf_x), jnp.full_like(obj_y, rf_y), jnp.full_like(eq_radii, rf_r), obj_x, obj_y, eq_radii)

        covered_area = jnp.sum(overlaps)
        background_area = jnp.maximum(0.0, rf_area - covered_area)

        mean_rgb = (jnp.sum(overlaps[:, None] * obj_colors, axis=0) + background_area * background_color
        ) / rf_area

        var_rgb = (
            jnp.sum(
                overlaps[:, None] * (obj_colors - mean_rgb) ** 2,
                axis=0,
            )
            + background_area * (background_color - mean_rgb) ** 2
        ) / rf_area

        return mean_rgb, jnp.maximum(var_rgb, 1e-6)

    means, variances = vmap(compute_single_rf)(rfs)

    return means, variances


def scene_to_jax(scene, rfs_list):
    ''' 
    INTERNAL HELPER
    Convert the Scene object and receptive field list into JAX arrays.
    '''
    rfs = jnp.array([[rf["x"], rf["y"], rf["r"]] for rf in rfs_list], dtype=jnp.float32)

    objects = []

    for obj in scene.objects:

        if hasattr(obj, "radius"):
            obj_type = 0
            size = obj.radius

        elif obj.__class__.__name__ == "Square":
            obj_type = 1
            size = obj.size

        elif obj.__class__.__name__ == "Triangle":
            obj_type = 2
            size = obj.size

        else:
            continue

        r, g, b = obj.color

        objects.append([obj.x, obj.y, size, obj_type, r, g, b,])

    objects = jnp.array(objects, dtype=jnp.float32)

    background = jnp.array(scene.background_color, dtype=jnp.float32)

    return rfs, objects, background

def jax_stats(scene, rfs_list):
    '''
    INTERNAL HELPER
    Convenience wrapper that converts a Scene to JAX arrays and
    computes receptive field statistics.
    Returns
    -------
    means : (N,3)
    variances : (N,3)
    '''
    rfs, objects, background = scene_to_jax(scene, rfs_list)

    means, variances = predict_rf_stats(rfs, objects, background)

    return means, variances

import numpy as np # Ensure numpy is imported at the top of your file
def predict_rf_stats_jax(scene, rfs_list):
    '''
    MAIN API 
    Wrapper to convert standard Python OOP scene objects into JAX arrays,
    run the jitted JAX function, and return standard dictionaries.
    '''
    means, variances = jax_stats(scene, rfs_list)

    # Format back into a standard list of dictionaries
    statistical_map = []
    for i, rf in enumerate(rfs_list):
        # Convert JAX arrays back to standard NumPy arrays for SciPy/Matplotlib compatibility
        mean_rgb = np.array(means[i])
        var_rgb = np.array(variances[i])
        
        statistical_map.append({
            "rf": rf,
            "rf_area": np.pi * rf["r"]**2,
            "pixel_count": 0, # Added to prevent KeyError in save_statistics_to_csv
            "mean": mean_rgb,
            "variance": var_rgb
        })

    return statistical_map

def rf_random(scene, rfs_list, key):
    '''
    MAIN API
    Sample one RGB value for every receptive field from the predicted
    Gaussian distribution.
    '''
    means, variances = jax_stats(scene, rfs_list)

    std = jnp.sqrt(jnp.maximum(variances, 1e-6))

    noise = normal(key, means.shape)

    return means + noise * std

def sample_image_at_rfs(img, rfs):
    '''
    INTERNAL HELPER
    Sample the RGB value at the center of each receptive field
    '''
    x = jnp.clip(
        jnp.round(rfs[:, 0]).astype(jnp.int32),
        0,
        img.shape[1] - 1,
    )
    y = jnp.clip(
        jnp.round(rfs[:, 1]).astype(jnp.int32),
        0,
        img.shape[0] - 1,
    )

    return img[y, x]

def rf_logpdf(img, scene, rfs_list):
    '''
    MAIN API
    Compute the log probability of an observed image given a scene.
    '''
    means, variances = jax_stats(scene, rfs_list)

    rfs, _, _ = scene_to_jax(scene, rfs_list)

    variances = jnp.maximum(variances, 1e-6)

    observed_rgb = sample_image_at_rfs(img, rfs)

    log_probs = (
        -0.5 * jnp.log(2.0 * jnp.pi * variances)
        -0.5 * ((observed_rgb - means) ** 2) / variances
    )

    return jnp.sum(log_probs)

if __name__ == "__main__":

    # Example receptive fields
    rfs = jnp.array([
        [100.0, 100.0, 30.0],
        [200.0, 150.0, 40.0],
        [300.0, 250.0, 50.0],
    ])

    '''
    Objects:
    [x, y, size, type, r, g, b]
    
    type:
        0 = circle
        1 = square
        2 = triangle
    '''
    objects = jnp.array([
        [100.0, 100.0, 25.0, 0.0, 1.0, 0.0, 0.0],
        [220.0, 170.0, 40.0, 1.0, 0.0, 1.0, 0.0],
        [280.0, 260.0, 35.0, 2.0, 0.0, 0.0, 1.0],
    ])

    background = jnp.array([1.0, 1.0, 1.0])

    stats = predict_rf_stats(rfs, objects, background)

    print("\nPredicted RF Statistics:\n")
    print(stats)