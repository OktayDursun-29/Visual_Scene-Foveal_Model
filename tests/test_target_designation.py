import unittest
from dataclasses import replace

import numpy as np

from foveal_scene import (
    Circle,
    DistanceTargetDesignator,
    SimulationState,
    VisualScene2D,
    empirical_attention_delta,
    update_after_attention,
)


class TargetDesignationTests(unittest.TestCase):
    def setUp(self):
        self.scene = VisualScene2D(width=200, height=100)
        self.reference = SimulationState(
            scene=self.scene,
            objects=(
                Circle(center=(20, 50), radius=5, fill="red"),
                Circle(center=(180, 50), radius=5, fill="blue"),
            ),
            rfs={},
        )
        self.designator = DistanceTargetDesignator.from_labeled_state(
            self.reference,
            target_indices=[0],
            distractor_indices=[1],
            position_scale=10_000,
        )

    def test_designates_target_like_objects_without_an_rfs_likelihood(self):
        state = replace(
            self.reference,
            objects=(
                Circle(center=(100, 20), radius=5, fill=(245, 10, 10)),
                Circle(center=(100, 80), radius=5, fill=(10, 10, 245)),
            ),
        )

        result = self.designator.designate(state)

        self.assertGreater(result.probability_for(0), 0.9)
        self.assertLess(result.probability_for(1), 0.1)

    def test_delta_pi_is_change_in_marginals_not_a_trace_weight(self):
        ambiguous = replace(
            self.reference,
            objects=(Circle(center=(100, 50), radius=5, fill=(128, 0, 128)),),
        )
        attended = replace(
            ambiguous,
            objects=(Circle(center=(100, 50), radius=5, fill=(250, 5, 5)),),
        )

        update = update_after_attention(self.designator, ambiguous, attended)

        self.assertEqual(update.delta_pi.shape, (1,))
        self.assertAlmostEqual(
            update.delta_pi[0],
            update.after.target_probabilities[0] - update.before.target_probabilities[0],
        )
        self.assertGreater(update.delta_pi[0], 0.4)

    def test_empirical_delta_averages_attention_outcomes(self):
        ambiguous = replace(
            self.reference,
            objects=(Circle(center=(100, 50), radius=5, fill=(128, 0, 128)),),
        )
        target_sample = replace(
            ambiguous,
            objects=(Circle(center=(100, 50), radius=5, fill="red"),),
        )
        distractor_sample = replace(
            ambiguous,
            objects=(Circle(center=(100, 50), radius=5, fill="blue"),),
        )

        estimate = empirical_attention_delta(
            self.designator, ambiguous, [target_sample, distractor_sample]
        )

        np.testing.assert_allclose(estimate, np.zeros(1), atol=0.02)


if __name__ == "__main__":
    unittest.main()
