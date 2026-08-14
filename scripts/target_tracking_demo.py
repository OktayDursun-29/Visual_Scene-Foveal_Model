"""Visible end-to-end demo of heuristic target designation and tracking."""

from dataclasses import replace
from pathlib import Path

from PIL import ImageDraw, ImageFont

from foveal_scene import (
    Circle,
    DistanceTargetDesignator,
    FoveatedObjectRenderer,
    SimulationState,
    SimulationStudy,
    VelocityMotionPrior,
    VisualScene2D,
)


def main() -> None:
    scene = VisualScene2D(width=640, height=360, background=(245, 245, 240))
    initial = SimulationState(
        scene=scene,
        objects=(
            Circle(
                center=(90, 95),
                radius=18,
                fill=(225, 55, 55),
                velocity=(18, 9),
            ),
            Circle(
                center=(530, 80),
                radius=18,
                fill=(55, 90, 225),
                velocity=(-12, 8),
            ),
            Circle(
                center=(470, 285),
                radius=18,
                fill=(45, 175, 90),
                velocity=(-10, -7),
            ),
        ),
        rfs={},
        fixation=(90, 95),
    )

    # Object 0 is the labeled target. A very large position scale makes color
    # identity dominate, so the target remains designated as it moves.
    designator = DistanceTargetDesignator.from_labeled_state(
        initial,
        target_indices=[0],
        distractor_indices=[1, 2],
        position_scale=10_000.0,
    )
    study = SimulationStudy(VelocityMotionPrior())
    renderer = FoveatedObjectRenderer(scene)
    state = initial
    frames = []

    for frame_number in range(16):
        designation = designator.designate(state)
        selected_offset = int(designation.target_probabilities.argmax())
        selected_index = designation.object_indices[selected_offset]
        selected = state.objects[selected_index]
        fixation = selected.center

        frame = renderer.render(state.objects, fixation=fixation)
        draw = ImageDraw.Draw(frame)
        font = ImageFont.load_default()
        for index, (obj, probability) in enumerate(
            zip(state.objects, designation.target_probabilities, strict=True)
        ):
            x, y = obj.center
            draw.text(
                (x - 31, y + obj.radius + 6),
                f"obj {index}: pi={probability:.2f}",
                fill=(15, 15, 15),
                font=font,
            )
        x, y = fixation
        draw.ellipse((x - 28, y - 28, x + 28, y + 28), outline=(255, 195, 0), width=4)
        draw.text(
            (12, 12),
            f"frame {frame_number:02d} | tracking object {selected_index}",
            fill=(15, 15, 15),
            font=font,
        )
        frames.append(frame)

        state = replace(state, fixation=fixation, observation=frame)
        state = study.time_step(state)

    output_dir = Path(__file__).resolve().parent.parent / "examples" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "target_tracking.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=350,
        loop=0,
    )
    print(f"Wrote target tracking animation to: {output_path}")


if __name__ == "__main__":
    main()
