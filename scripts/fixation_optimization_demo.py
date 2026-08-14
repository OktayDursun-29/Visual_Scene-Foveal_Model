"""Optimize a next fixation from a central, stationary starting point."""

import jax.numpy as jnp

from foveal_scene import resolve_next_fixation_sgd, total_fixation_loss


def main() -> None:
    # Shape: (objects, particle samples, horizon states, x/y bearing).
    relevant = jnp.tile(jnp.array([[[[110.0, 30.0]]]]), (1, 32, 3, 1))
    distractor = jnp.tile(jnp.array([[[[-170.0, -80.0]]]]), (1, 32, 3, 1))
    target_samples = jnp.concatenate((distractor, relevant), axis=0)
    task_relevance = jnp.array([0.0, 5.0])  # second object is much more relevant
    f_t = jnp.array([0.0, 0.0])
    v_t = jnp.array([0.0, 0.0])

    f_next = resolve_next_fixation_sgd(f_t, v_t, target_samples, task_relevance)
    # The test weights isolate the relevant target when displaying improvement.
    relevant_only = jnp.array([0.0, 1.0])
    start_loss = total_fixation_loss(f_t, f_t, v_t, target_samples, relevant_only)
    end_loss = total_fixation_loss(f_next, f_t, v_t, target_samples, relevant_only)
    print(f"Start fixation: {tuple(float(x) for x in f_t)}")
    print(f"Optimized fixation: {tuple(float(x) for x in f_next)}")
    print(f"Loss: {float(start_loss):.4f} -> {float(end_loss):.4f}")


if __name__ == "__main__":
    main()
