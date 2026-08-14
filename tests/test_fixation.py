import unittest

import jax.numpy as jnp

from foveal_scene import (
    foveal_coverage_loss,
    movement_cost,
    resolve_next_fixation_sgd,
    total_fixation_loss,
)


class FixationOptimizationTests(unittest.TestCase):
    def test_coverage_is_lower_loss_near_a_relevant_object(self):
        samples = jnp.array([[[[100.0, 0.0]], [[102.0, -1.0]]]])
        weights = jnp.array([1.0])

        near_loss = foveal_coverage_loss(jnp.array([100.0, 0.0]), samples, weights)
        far_loss = foveal_coverage_loss(jnp.array([0.0, 0.0]), samples, weights)

        self.assertLess(float(near_loss), float(far_loss))

    def test_movement_cost_rewards_continuing_the_current_velocity(self):
        previous = jnp.array([0.0, 0.0])
        velocity = jnp.array([10.0, 0.0])
        smooth = movement_cost(jnp.array([10.0, 0.0]), previous, velocity)
        sharp_turn = movement_cost(jnp.array([0.0, 10.0]), previous, velocity)

        self.assertLess(float(smooth), float(sharp_turn))

    def test_centered_stationary_fixation_optimizes_toward_relevant_samples(self):
        # Two objects with 32 identical predicted particles over a three-step horizon.
        relevant = jnp.tile(jnp.array([[[[110.0, 30.0]]]]), (1, 32, 3, 1))
        distractor = jnp.tile(jnp.array([[[[-170.0, -80.0]]]]), (1, 32, 3, 1))
        samples = jnp.concatenate((distractor, relevant), axis=0)
        f_t = jnp.array([0.0, 0.0])
        v_t = jnp.array([0.0, 0.0])
        task_relevance = jnp.array([0.0, 5.0])
        weights = jnp.array([0.0, 1.0])

        f_next = resolve_next_fixation_sgd(f_t, v_t, samples, task_relevance)
        start_loss = total_fixation_loss(f_t, f_t, v_t, samples, weights)
        end_loss = total_fixation_loss(f_next, f_t, v_t, samples, weights)

        self.assertGreater(float(f_next[0]), 0.0)
        self.assertGreater(float(f_next[1]), 0.0)
        self.assertLess(float(end_loss), float(start_loss))


if __name__ == "__main__":
    unittest.main()
