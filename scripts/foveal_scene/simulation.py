"""Stateful simulation utilities built on the static foveated renderer.

The original rendering API remains deliberately static.  This module adds the
generative/dynamics layer: objects carry velocities, motion priors generate the
next object positions, a gaze controller chooses fixation, and an observer
renders the resulting scene.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Protocol, Sequence

import numpy as np
from PIL import Image

from .objects import MovableSceneObject, Point
from .renderer import FoveatedObjectRenderer
from .scene import VisualScene2D


class MotionPrior(Protocol):
    """Samples object positions for one transition of the generative model."""

    def sample(
        self,
        *,
        scene: VisualScene2D,
        objects: Sequence[MovableSceneObject],
        rfs: object,
        time: float,
        dt: float,
        rng: np.random.Generator,
    ) -> tuple[MovableSceneObject, ...]: ...


class PositionPredictor(Protocol):
    """Model interface for predicted object centers/anchors from RFs."""

    def predict_positions(
        self,
        objects: Sequence[MovableSceneObject], rfs: object, time: float
    ) -> Mapping[int, Point] | Sequence[Point]: ...


class GazeController(Protocol):
    """Chooses where the observer looks after a scene transition."""

    def choose_fixation(
        self,
        *,
        objects: Sequence[MovableSceneObject],
        rfs: object,
        time: float,
        previous_fixation: Point | None,
    ) -> Point: ...


@dataclass(frozen=True)
class VelocityMotionPrior:
    """Constant-velocity motion with optional Gaussian velocity noise."""

    velocity_noise_std: float = 0.0

    def sample(self, *, scene, objects, rfs, time, dt, rng):  # type: ignore[no-untyped-def]
        if dt <= 0:
            raise ValueError("dt must be positive")
        if self.velocity_noise_std < 0:
            raise ValueError("velocity_noise_std must be non-negative")
        moved: list[MovableSceneObject] = []
        for obj in objects:
            noise = rng.normal(0.0, self.velocity_noise_std, size=2)
            vx, vy = obj.velocity
            moved.append(obj.translated(((vx + noise[0]) * dt, (vy + noise[1]) * dt)))
        return tuple(moved)


@dataclass(frozen=True)
class PredictedPositionPrior:
    """Use a generative model's RF-conditioned position predictions as motion."""

    predictor: PositionPredictor

    def sample(self, *, scene, objects, rfs, time, dt, rng):  # type: ignore[no-untyped-def]
        if dt <= 0:
            raise ValueError("dt must be positive")
        predictions = self.predictor.predict_positions(objects, rfs, time)
        positions = (
            [predictions[index] for index in range(len(objects))]
            if isinstance(predictions, Mapping)
            else list(predictions)
        )
        if len(positions) != len(objects):
            raise ValueError("predictor must return one position for every object")

        predicted: list[MovableSceneObject] = []
        for obj, target in zip(objects, positions, strict=True):
            x, y = _anchor(obj)
            dx, dy = target[0] - x, target[1] - y
            # Velocity becomes an explicit part of the next state as well as
            # determining the new position.
            predicted.append(replace(obj.translated((dx, dy)), velocity=(dx / dt, dy / dt)))
        return tuple(predicted)


@dataclass(frozen=True)
class CompositeMotionPrior:
    """Compose independent motion-prior components in their listed order."""

    priors: Sequence[MotionPrior]

    def sample(self, *, scene, objects, rfs, time, dt, rng):  # type: ignore[no-untyped-def]
        result = tuple(objects)
        for prior in self.priors:
            result = prior.sample(
                scene=scene, objects=result, rfs=rfs, time=time, dt=dt, rng=rng
            )
        return result


@dataclass(frozen=True)
class FixedGazeController:
    fixation: Point

    def choose_fixation(self, *, objects, rfs, time, previous_fixation):  # type: ignore[no-untyped-def]
        return self.fixation


@dataclass(frozen=True)
class ScanpathGazeController:
    """A deterministic looking-around policy that cycles through fixations."""

    fixations: Sequence[Point]

    def choose_fixation(self, *, objects, rfs, time, previous_fixation):  # type: ignore[no-untyped-def]
        if not self.fixations:
            raise ValueError("fixations must not be empty")
        return self.fixations[int(np.floor(time)) % len(self.fixations)]


@dataclass(frozen=True)
class SimulationState:
    """Complete state of one simulated scene at a single time point."""

    scene: VisualScene2D
    objects: tuple[MovableSceneObject, ...]
    rfs: object
    time: float = 0.0
    fixation: Point | None = None
    observation: Image.Image | None = None


@dataclass
class SimulationStudy:
    """Run an RF-conditioned, foveated generative scene simulation."""

    motion_prior: MotionPrior
    gaze_controller: GazeController | None = None
    observer: FoveatedObjectRenderer | None = None
    dt: float = 1.0
    rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.rng is None:
            self.rng = np.random.default_rng()

    def time_step(self, scene_t: SimulationState, rfs_t: object | None = None) -> SimulationState:
        """Sample motion, observe the new scene, and return its next state."""
        rfs = scene_t.rfs if rfs_t is None else rfs_t
        objects = self.motion_prior.sample(
            scene=scene_t.scene,
            objects=scene_t.objects,
            rfs=rfs,
            time=scene_t.time,
            dt=self.dt,
            rng=self.rng,
        )
        new_time = scene_t.time + self.dt
        fixation = (
            self.gaze_controller.choose_fixation(
                objects=objects,
                rfs=rfs,
                time=new_time,
                previous_fixation=scene_t.fixation,
            )
            if self.gaze_controller is not None
            else scene_t.fixation
        )
        observer = self.observer or FoveatedObjectRenderer(scene_t.scene)
        observation = observer.render(objects, fixation=fixation)
        return SimulationState(
            scene=scene_t.scene,
            objects=objects,
            rfs=rfs,
            time=new_time,
            fixation=fixation,
            observation=observation,
        )

    def run(
        self,
        initial_state: SimulationState,
        steps: int,
        rfs_by_step: Sequence[object] | None = None,
    ) -> tuple[SimulationState, ...]:
        """Run a simulation study and return the initial state plus every step.

        ``rfs_by_step[i]`` is used for transition ``i``.  When omitted, each
        transition uses the RFs stored in the preceding state.
        """
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if rfs_by_step is not None and len(rfs_by_step) != steps:
            raise ValueError("rfs_by_step must contain exactly one entry per step")
        states = [initial_state]
        for index in range(steps):
            rfs = None if rfs_by_step is None else rfs_by_step[index]
            states.append(self.time_step(states[-1], rfs))
        return tuple(states)


def time_step(
    scene_t: SimulationState,
    rfs_t: object | None,
    *,
    motion_prior: MotionPrior,
    gaze_controller: GazeController | None = None,
    observer: FoveatedObjectRenderer | None = None,
    dt: float = 1.0,
    rng: np.random.Generator | None = None,
) -> SimulationState:
    """One-shot convenience wrapper around :meth:`SimulationStudy.time_step`."""
    return SimulationStudy(
        motion_prior=motion_prior,
        gaze_controller=gaze_controller,
        observer=observer,
        dt=dt,
        rng=rng,
    ).time_step(scene_t, rfs_t)


def _anchor(obj: MovableSceneObject) -> Point:
    """Return the anchor used by prediction priors for built-in drawable objects."""
    if hasattr(obj, "center"):
        return obj.center  # type: ignore[return-value]
    if hasattr(obj, "xy"):
        x0, y0, x1, y1 = obj.xy  # type: ignore[attr-defined]
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    if hasattr(obj, "points"):
        points = obj.points  # type: ignore[attr-defined]
        if not points:
            raise ValueError("cannot predict the position of an object with no points")
        return points[0]
    raise TypeError(f"{type(obj).__name__} needs a center, xy, or points anchor")
