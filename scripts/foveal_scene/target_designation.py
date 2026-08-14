"""Target designation without a random-finite-set observation likelihood.

The module operates on an object-level representation. A ``Segmenter`` is the
boundary between receptive-field perception and decision making; the built-in
segmenter exposes objects already present in ``SimulationState`` and can later
be replaced by an RF/image segmenter.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Protocol, Sequence

import numpy as np

from .objects import MovableSceneObject, Point
from .simulation import SimulationState


@dataclass(frozen=True)
class ObjectDescriptor:
    """Decision-level description of one segmented scene object."""

    object_index: int
    position: Point
    color: tuple[float, float, float]
    size: float
    shape: str


class Segmenter(Protocol):
    """Converts a perceptual state into an unordered object representation."""

    def segment(self, state: SimulationState) -> tuple[ObjectDescriptor, ...]: ...


@dataclass(frozen=True)
class SceneObjectSegmenter:
    """Adapter for simulations whose latent objects are directly available.

    This is a baseline, not an image/RF segmentation model. Keeping it behind
    ``Segmenter`` makes that modeling shortcut explicit.
    """

    def segment(self, state: SimulationState) -> tuple[ObjectDescriptor, ...]:
        return tuple(_describe(index, obj) for index, obj in enumerate(state.objects))


@dataclass(frozen=True)
class DesignationResult:
    """Marginal probability that each object is a target."""

    object_indices: tuple[int, ...]
    target_probabilities: np.ndarray

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.target_probabilities, dtype=float)
        if probabilities.shape != (len(self.object_indices),):
            raise ValueError("target_probabilities must have one entry per object")
        if not np.all(np.isfinite(probabilities)) or np.any(
            (probabilities < 0.0) | (probabilities > 1.0)
        ):
            raise ValueError("target probabilities must be finite and in [0, 1]")
        object.__setattr__(self, "target_probabilities", probabilities)

    def probability_for(self, object_index: int) -> float:
        try:
            offset = self.object_indices.index(object_index)
        except ValueError as error:
            raise KeyError(object_index) from error
        return float(self.target_probabilities[offset])


@dataclass(frozen=True)
class DistanceTargetDesignator:
    """Classify objects by distance to target and distractor exemplars.

    Distances combine normalized position, RGB, size, and shape. The signed
    distance margin is mapped to a target probability, so no RFS likelihood
    over receptive-field responses is required.
    """

    target_exemplars: tuple[ObjectDescriptor, ...]
    distractor_exemplars: tuple[ObjectDescriptor, ...]
    position_scale: float = 100.0
    color_scale: float = 255.0
    size_scale: float = 25.0
    shape_mismatch_cost: float = 1.0
    temperature: float = 0.25
    target_prior: float = 0.5

    def __post_init__(self) -> None:
        if not self.target_exemplars or not self.distractor_exemplars:
            raise ValueError("at least one target and distractor exemplar are required")
        for name in ("position_scale", "color_scale", "size_scale", "temperature"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.shape_mismatch_cost < 0:
            raise ValueError("shape_mismatch_cost must be non-negative")
        if not 0.0 < self.target_prior < 1.0:
            raise ValueError("target_prior must be strictly between zero and one")

    @classmethod
    def from_labeled_state(
        cls,
        state: SimulationState,
        *,
        target_indices: Sequence[int],
        distractor_indices: Sequence[int] | None = None,
        segmenter: Segmenter | None = None,
        **kwargs: float,
    ) -> "DistanceTargetDesignator":
        """Create exemplar sets from known labels in a reference state."""

        descriptors = (segmenter or SceneObjectSegmenter()).segment(state)
        by_index = {item.object_index: item for item in descriptors}
        target_set = set(target_indices)
        distractor_set = (
            set(by_index).difference(target_set)
            if distractor_indices is None
            else set(distractor_indices)
        )
        if target_set & distractor_set:
            raise ValueError("target and distractor indices must be disjoint")
        missing = (target_set | distractor_set).difference(by_index)
        if missing:
            raise IndexError(f"unknown object indices: {sorted(missing)}")
        return cls(
            target_exemplars=tuple(by_index[index] for index in sorted(target_set)),
            distractor_exemplars=tuple(by_index[index] for index in sorted(distractor_set)),
            **kwargs,
        )

    def designate(
        self,
        state: SimulationState,
        *,
        segmenter: Segmenter | None = None,
    ) -> DesignationResult:
        descriptors = (segmenter or SceneObjectSegmenter()).segment(state)
        probabilities = np.asarray(
            [self.probability(descriptor) for descriptor in descriptors], dtype=float
        )
        return DesignationResult(
            tuple(descriptor.object_index for descriptor in descriptors), probabilities
        )

    def probability(self, descriptor: ObjectDescriptor) -> float:
        target_distance = min(
            self._distance(descriptor, exemplar) for exemplar in self.target_exemplars
        )
        distractor_distance = min(
            self._distance(descriptor, exemplar) for exemplar in self.distractor_exemplars
        )
        prior_log_odds = log(self.target_prior / (1.0 - self.target_prior))
        log_odds = prior_log_odds + (distractor_distance - target_distance) / self.temperature
        if log_odds >= 0:
            return float(1.0 / (1.0 + np.exp(-log_odds)))
        exponential = np.exp(log_odds)
        return float(exponential / (1.0 + exponential))

    def _distance(self, left: ObjectDescriptor, right: ObjectDescriptor) -> float:
        position = np.linalg.norm(np.subtract(left.position, right.position)) / self.position_scale
        color = np.linalg.norm(np.subtract(left.color, right.color)) / self.color_scale
        size = abs(left.size - right.size) / self.size_scale
        shape = self.shape_mismatch_cost if left.shape != right.shape else 0.0
        return float(np.sqrt(position**2 + color**2 + size**2 + shape**2))


@dataclass(frozen=True)
class AttentionUpdate:
    """Target marginals before/after attention and their actual difference."""

    before: DesignationResult
    after: DesignationResult
    delta_pi: np.ndarray


def update_after_attention(
    designator: DistanceTargetDesignator,
    before_state: SimulationState,
    attended_state: SimulationState,
    *,
    segmenter: Segmenter | None = None,
) -> AttentionUpdate:
    """Compute ``delta_pi = pi(attended_state) - pi(before_state)``."""

    before = designator.designate(before_state, segmenter=segmenter)
    after = designator.designate(attended_state, segmenter=segmenter)
    if before.object_indices != after.object_indices:
        raise ValueError("attention update must preserve object identities")
    return AttentionUpdate(before, after, after.target_probabilities - before.target_probabilities)


def empirical_attention_delta(
    designator: DistanceTargetDesignator,
    state: SimulationState,
    attended_states: Sequence[SimulationState],
    *,
    segmenter: Segmenter | None = None,
) -> np.ndarray:
    """Monte Carlo estimate of expected ``delta_pi`` under an attention action."""

    if not attended_states:
        raise ValueError("attended_states must contain at least one sample")
    deltas = [
        update_after_attention(designator, state, sample, segmenter=segmenter).delta_pi
        for sample in attended_states
    ]
    return np.mean(np.stack(deltas), axis=0)


def _describe(index: int, obj: MovableSceneObject) -> ObjectDescriptor:
    position = _anchor(obj)
    color = _rgb(getattr(obj, "fill", (0, 0, 0)))
    if hasattr(obj, "center"):
        size = float(obj.radius) * 2.0  # type: ignore[attr-defined]
    elif hasattr(obj, "xy"):
        x0, y0, x1, y1 = obj.xy  # type: ignore[attr-defined]
        size = float(np.sqrt(abs((x1 - x0) * (y1 - y0))))
    elif hasattr(obj, "points"):
        points = np.asarray(obj.points, dtype=float)  # type: ignore[attr-defined]
        size = float(np.linalg.norm(np.ptp(points, axis=0))) if len(points) else 0.0
    else:
        size = 0.0
    return ObjectDescriptor(index, position, color, size, type(obj).__name__.lower())


def _anchor(obj: MovableSceneObject) -> Point:
    if hasattr(obj, "center"):
        x, y = obj.center  # type: ignore[attr-defined]
        return (float(x), float(y))
    if hasattr(obj, "xy"):
        x0, y0, x1, y1 = obj.xy  # type: ignore[attr-defined]
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    if hasattr(obj, "points"):
        points = obj.points  # type: ignore[attr-defined]
        if not points:
            raise ValueError("cannot designate an object with no points")
        coordinates = np.asarray(points, dtype=float)
        x, y = np.mean(coordinates, axis=0)
        return (float(x), float(y))
    raise TypeError(f"{type(obj).__name__} needs a center, xy, or points anchor")


def _rgb(color: object) -> tuple[float, float, float]:
    if isinstance(color, str):
        from PIL import ImageColor

        color = ImageColor.getrgb(color)
    values = tuple(float(value) for value in color)  # type: ignore[union-attr]
    if len(values) < 3:
        raise ValueError("object colors must have at least three channels")
    return values[:3]  # type: ignore[return-value]
