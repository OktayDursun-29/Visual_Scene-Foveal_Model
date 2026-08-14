# Visual Scene–Foveal Model

The project has one maintained implementation of each component. All code and
runnable entry points are under `scripts/`; tests are under `tests/`; generated
files stay under `img/`, `results/`, and `examples/output/`.

## Code structure

- `scripts/foveal_scene/`: scene objects, two-level foveation, simulation,
  fixation optimization, and target designation.
- `scripts/ReceptiveFieldsJAX.py`: RF layout, JAX RF statistics, sampling, and
  the Python/Julia array bridge.
- `scripts/Apply_ReceptiveFields.py`: the single RF analysis runner.
- `scripts/GenJLModel.jl`: the active Gen.jl motion and observation model.
- `scripts/Scene.py`: regenerates the original scene.
- `scripts/Foveal_Image.py`: regenerates the two-level foveated image.
- `scripts/target_tracking_demo.py`: runs target designation and writes its GIF.
- `scripts/fixation_optimization_demo.py`: runs the fixation example.

## Setup

From the repository root:

```powershell
& .\scripts\setup.ps1
```

## Run

```powershell
# Tests
& .\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q

# Original scene and its two-level foveated result
& .\.venv\Scripts\python.exe .\scripts\Scene.py
& .\.venv\Scripts\python.exe .\scripts\Foveal_Image.py

# Target designation/tracking
& .\.venv\Scripts\python.exe .\scripts\target_tracking_demo.py

# Fixation example
& .\.venv\Scripts\python.exe .\scripts\fixation_optimization_demo.py

# Receptive-field analysis
& .\.venv\Scripts\python.exe .\scripts\Apply_ReceptiveFields.py
```

## Gen.jl

After configuring Julia packages `Gen` and `PyCall` against this workspace's
Python environment:

```powershell
julia tests/test_pycall_rf.jl
julia scripts/GenJLModel.jl
```

The Gen.jl trajectory is written as the single animated file
`results/genjl_states/trajectory_states.svg`. Faint red circles in the SVG
show the sampled states while the solid red circle animates through them.
