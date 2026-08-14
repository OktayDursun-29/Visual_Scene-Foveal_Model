import unittest

import numpy as np

from foveal_scene import (
    Circle,
    FixedGazeController,
    PredictedPositionPrior,
    SimulationState,
    SimulationStudy,
    VelocityMotionPrior,
    VisualScene2D,
)


class ConstantPredictor:
    def predict_positions(self, objects, rfs, time):
        return [(rfs["x"], rfs["y"])]


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.scene = VisualScene2D(width=100, height=80)

    def test_velocity_prior_moves_objects_and_observes_new_state(self):
        state = SimulationState(
            scene=self.scene,
            objects=(Circle(center=(10, 20), radius=3, fill="red", velocity=(4, -2)),),
            rfs={"unused": True},
        )
        study = SimulationStudy(
            motion_prior=VelocityMotionPrior(),
            gaze_controller=FixedGazeController((14, 18)),
            rng=np.random.default_rng(5),
        )

        next_state = study.time_step(state)

        self.assertEqual(next_state.time, 1.0)
        self.assertEqual(next_state.objects[0].center, (14, 18))
        self.assertEqual(next_state.objects[0].velocity, (4, -2))
        self.assertEqual(next_state.fixation, (14, 18))
        self.assertEqual(next_state.observation.size, (100, 80))

    def test_predictions_conditioned_on_rfs_set_position_and_velocity(self):
        state = SimulationState(
            scene=self.scene,
            objects=(Circle(center=(2, 3), radius=3, fill="red"),),
            rfs={"x": 12, "y": 9},
        )
        study = SimulationStudy(
            motion_prior=PredictedPositionPrior(ConstantPredictor()),
            dt=2,
            rng=np.random.default_rng(0),
        )

        next_state = study.time_step(state, {"x": 12, "y": 9})

        self.assertEqual(next_state.objects[0].center, (12, 9))
        self.assertEqual(next_state.objects[0].velocity, (5, 3))

    def test_run_returns_a_time_indexed_study(self):
        state = SimulationState(
            scene=self.scene,
            objects=(Circle(center=(0, 0), radius=3, fill="red", velocity=(1, 0)),),
            rfs={},
        )
        states = SimulationStudy(VelocityMotionPrior(), rng=np.random.default_rng(0)).run(
            state, steps=2
        )

        self.assertEqual([item.time for item in states], [0.0, 1.0, 2.0])
        self.assertEqual(states[-1].objects[0].center, (2, 0))


if __name__ == "__main__":
    unittest.main()
