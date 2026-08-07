#!/usr/bin/env python3

# INVALID 
# GEN JAX VERSION OF GenJLModel.jl

"""
GenerativeModel.py: Dynamic probabilistic visual scene model.
Pipeline: Scene_t -> Motion Prior -> Scene_t+1 -> RF Obs -> RGB samples.
"""
import copy
import jax
import jax.numpy as jnp
import genjax
from genjax import gen, normal
from jax.random import split
from Scene import Scene, Circle, Square, Triangle
from ReceptiveFields_gen import ReceptiveFields
from ReceptiveFieldsJAX import scene_to_jax
import matplotlib.pyplot as plt

# Configuration & Custom RF Distribution
IMAGE_WIDTH, IMAGE_HEIGHT = 800, 800
MAX_SPEED, ACCEL_STD = 20.0, 1.0
BACKGROUND = (255, 255, 255)
receptive_fields = ReceptiveFields()

# Helper Functions
def clamp(value, minimum, maximum):
    """Clamp a value between minimum and maximum."""
    return max(minimum, min(value, maximum))

def clone_scene(scene):
    """Deep copy scene so previous states are not modified."""
    return copy.deepcopy(scene)

def clone_object(obj):
    """Deep copy an object."""
    return copy.deepcopy(obj)

def get_objects(scene):
    """Support different Scene implementations (objects or shapes list)."""
    if hasattr(scene, "objects"):
        return scene.objects
    elif hasattr(scene, "shapes"):
        return scene.shapes
    raise AttributeError("Scene must contain objects or shapes list")

def ensure_velocity(obj):
    if not hasattr(obj, "vx"):
        obj.vx = 8.0
    if not hasattr(obj, "vy"):
        obj.vy = 4.0

def scene_to_genjax(scene, rfs):
    """Convert Python scene into JAX tensors."""
    rfs_jax, objects, background = scene_to_jax(scene, rfs)
    return (objects, background), rfs_jax

# JAX Motion Model
def clamp_jax(value, minimum, maximum):
    """JAX-compatible clamp."""
    return jnp.clip(value, minimum, maximum)

@gen
def motion_prior(scene):
    """
    Samples next scene state:
    - moving objects receive acceleration
    - stationary objects remain fixed
    """
    new_scene = clone_scene(scene)
    objects = get_objects(new_scene)

    for i, obj in enumerate(objects):
        # Skip objects marked as stationary
        if getattr(obj, "stationary", False):
            continue

        ax = normal(0.0, ACCEL_STD) @ f"ax_{i}"
        ay = normal(0.0, ACCEL_STD) @ f"ay_{i}"

        update_velocity(obj, ax, ay)
        update_position(obj)

    return new_scene

@gen
def time_step(prev_state, rfs):
    new_state = motion_prior(prev_state) @ "motion"
    # RF observation temporarily disabled until valid RFs are provided
    return new_state

# Scan across time
time_step_scan = genjax.scan(n=5)(time_step)

def simulate_scene(initial_scene, rfs, steps=10):
    """Generate JAX scene states."""
    key = jax.random.PRNGKey(0)
    current_state = initial_scene
    rfs_jax = rfs
    states = []

    for t in range(steps):
        key, subkey = split(key)
        result = time_step.simulate(subkey, (current_state, rfs_jax))
        current_state = result.retval
        states.append(current_state)

    return states

def jax_to_scene(state):
    """Convert JAX state back into Scene object."""
    objects, background = state
    scene = Scene(background_color=tuple(int(x) for x in background))

    for obj in objects:
        x, y, size = float(obj[0]), float(obj[1]), float(obj[2])
        color = (int(obj[4]), int(obj[5]), int(obj[6]))
        shape_type = int(obj[3])

        if shape_type == 0:
            shape = Circle(x=x, y=y, radius=size, color=color)
        elif shape_type == 1:
            shape = Square(x=x, y=y, size=size, color=color)
        else:
            shape = Triangle(x=x, y=y, size=size, color=color)

        get_objects(scene).append(shape)

    return scene

def render_states(states):
    fig, axes = plt.subplots(1, len(states), figsize=(15, 3))

    for t, scene in enumerate(states):
        image = scene.render_to_image(IMAGE_WIDTH, IMAGE_HEIGHT)
        axes[t].imshow(image)
        axes[t].set_title(f"State {t}")
        axes[t].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Running Generative Model...")

    scene = Scene(background_color=BACKGROUND)

    # Moving red circle
    moving_circle = Circle(x=300, y=400, radius=30, color=(255, 0, 0))

    # Stationary blue circle
    stationary_circle = Circle(x=500, y=400, radius=30, color=(0, 0, 255), stationary=True)

    scene.add_object(moving_circle)
    scene.add_object(stationary_circle)

    rfs = []
    states = simulate_scene(scene, rfs, steps=10)
    render_states([jax_to_scene(s) for s in states])

    print("Simulation complete!")