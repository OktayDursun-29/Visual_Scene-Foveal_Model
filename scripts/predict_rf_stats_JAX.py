#!/usr/bin/env python3

import jax
import jax.numpy as jnp
from jax import jit, vmap


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

    alpha = jnp.arccos(
        jnp.clip((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1 + eps), -1.0, 1.0)
    )

    beta = jnp.arccos(
        jnp.clip((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2 + eps), -1.0, 1.0)
    )

    term = ((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))

    lens_area = (r1 ** 2 * alpha + r2 ** 2 * beta - 0.5 * jnp.sqrt(jnp.maximum(term, 0.0)))

    return jnp.where(
        no_overlap,
        0.0,
        jnp.where(
            contained,
            contained_area,
            lens_area,
        ),
    )


@jit
def predict_rf_stats(rfs, objects, background_color):
    
    # Args:
    #   rfs: (N_RF,3) -> [x, y, r]
    #  objects: (N_OBJ,7) -> [x, y, size, type, r, g, b]

    obj_x = objects[:, 0]
    obj_y = objects[:, 1]
    obj_size = objects[:, 2]
    obj_type = objects[:, 3]
    obj_colors = objects[:, 4:7]

    # Equivalent radii

    circle_radius = obj_size

    square_radius = jnp.sqrt((obj_size ** 2) / jnp.pi)

    triangle_radius = jnp.sqrt(((jnp.sqrt(3.0) / 4.0) * obj_size ** 2) / jnp.pi)

    eq_radii = jnp.where(
        obj_type == 0,
        circle_radius,
        jnp.where(
            obj_type == 1,
            square_radius,
            triangle_radius,
        ),
    )

    def compute_single_rf(rf_row):

        rf_x, rf_y, rf_r = rf_row
        rf_area = jnp.pi * rf_r ** 2

        overlaps = vmap(circle_circle_intersection)(
            jnp.full_like(obj_x, rf_x),
            jnp.full_like(obj_y, rf_y),
            jnp.full_like(eq_radii, rf_r),
            obj_x,
            obj_y,
            eq_radii,
        )

        covered_area = jnp.sum(overlaps)
        background_area = jnp.maximum(0.0, rf_area - covered_area)

        mean_rgb = (
            jnp.sum(overlaps[:, None] * obj_colors, axis=0)
            + background_area * background_color
        ) / rf_area

        var_rgb = (
            jnp.sum(
                overlaps[:, None] * (obj_colors - mean_rgb) ** 2,
                axis=0,
            )
            + background_area * (background_color - mean_rgb) ** 2
        ) / rf_area

        return jnp.concatenate(
            [mean_rgb, jnp.maximum(var_rgb, 1e-6)]
        )

    return vmap(compute_single_rf)(rfs)