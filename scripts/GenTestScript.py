import jax
from genjax import gen
from ReceptiveFields_gen import ReceptiveFields
from Scene import Scene, Circle
from ReceptiveFields import generate_receptive_fields

# Create a simple test scene
scene = Scene(
    background_color=(1.0, 1.0, 1.0)
)

scene.add_object(
    Circle(
        x=100,
        y=100,
        radius=25,
        color=(1.0, 0.0, 0.0)
    )
)

# Create receptive fields
rfs = generate_receptive_fields(
    image_shape=(512, 512),
    fixation=(256, 256),
    base_radius=10,
    growth_rate=1.5,
    overlap_density=0.5
)

# Create GenJAX distribution
receptive_fields = ReceptiveFields()

# Define generative model
@gen
def model(scene, rfs):

    x = receptive_fields(scene, rfs) @ "observed"

    return x

# Run simulation
key = jax.random.key(0)

from ReceptiveFieldsJAX import scene_to_jax

rfs_jax, objects_jax, background_jax = scene_to_jax(
    scene,
    rfs
)

scene_jax = (
    objects_jax,
    background_jax
)

trace = model.simulate(
    key,
    (scene_jax, rfs_jax)
)

print("\n=== GenJAX Simulation Successful ===\n")

print(f"Model: {trace.gen_fn.__class__.__name__}")

observed = trace.subtraces["observed"]

print(f"Distribution: {observed.gen_fn.__class__.__name__}")
print(f"Number of receptive fields: {observed.value.shape[0]}")
print(f"Channels per RF: {observed.value.shape[1]}")
print(f"Log Probability: {float(observed.score):.4f}")

print("\nFirst 5 sampled RGB values:\n")
print(observed.value[:5])