from pathlib import Path

from PIL import Image

from foveal_scene import BasicFovealModel, FovealConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "img" / "visual_scene.png"
OUTPUT_PATH = PROJECT_ROOT / "img" / "foveated_scene.png"


def main() -> None:
    image = Image.open(INPUT_PATH)
    model = BasicFovealModel(FovealConfig(fovea_radius=40, blur_radii=(0.0, 10.0)))
    foveated_image = model(image, fixation=(128, 128))
    foveated_image.save(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
