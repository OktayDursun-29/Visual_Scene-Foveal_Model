"""Simple 2D scene rendering and foveal image transforms."""

from .fovea import BasicFovealModel, FovealConfig
from .fixation import (
    foveal_coverage_loss,
    movement_cost,
    resolve_next_fixation_sgd,
    total_fixation_loss,
)
from .objects import Circle, Line, MovableSceneObject, Polygon, Rectangle, SceneObject
from .renderer import FoveatedObjectRenderer
from .scene import VisualScene2D
from .simulation import (
    CompositeMotionPrior,
    FixedGazeController,
    GazeController,
    PositionPredictor,
    PredictedPositionPrior,
    ScanpathGazeController,
    SimulationState,
    SimulationStudy,
    VelocityMotionPrior,
    time_step,
)
from .target_designation import (
    AttentionUpdate,
    DesignationResult,
    DistanceTargetDesignator,
    ObjectDescriptor,
    SceneObjectSegmenter,
    Segmenter,
    empirical_attention_delta,
    update_after_attention,
)

__all__ = [
    "AttentionUpdate",
    "BasicFovealModel",
    "Circle",
    "DesignationResult",
    "DistanceTargetDesignator",
    "empirical_attention_delta",
    "FovealConfig",
    "foveal_coverage_loss",
    "FoveatedObjectRenderer",
    "Line",
    "MovableSceneObject",
    "movement_cost",
    "ObjectDescriptor",
    "CompositeMotionPrior",
    "FixedGazeController",
    "GazeController",
    "PositionPredictor",
    "PredictedPositionPrior",
    "Polygon",
    "Rectangle",
    "SceneObject",
    "ScanpathGazeController",
    "SimulationState",
    "SimulationStudy",
    "SceneObjectSegmenter",
    "Segmenter",
    "resolve_next_fixation_sgd",
    "total_fixation_loss",
    "VelocityMotionPrior",
    "VisualScene2D",
    "time_step",
    "update_after_attention",
]
