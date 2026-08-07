# Visual_Scene-Foveal_Model

## Gen.jl dynamics model

`scripts/GenJLModel.jl` is the active dynamic model.  It implements the
motion prior and time-step model in Gen.jl, while `ReceptiveFieldsJAX.py`
continues to compute receptive-field means and variances through PyCall.
The old GenJAX files are retained as experiments; they are not required for
the Julia workflow.

Install Julia, then create a Julia environment with the required packages:

```julia
using Pkg
Pkg.add(["Gen", "PyCall"])
ENV["PYTHON"] = raw"C:\\path\\to\\the\\Python\\environment\\python.exe"
Pkg.build("PyCall")
```

The selected Python environment must have the existing project dependencies
installed (`jax`, `jaxlib`, and `numpy`).  Close and restart Julia after
building PyCall.  From the repository root, first validate the bridge and
then sample/render a trajectory:

```powershell
julia scripts/test_pycall_rf.jl
julia scripts/GenJLModel.jl
```

The second command writes five sampled scene states to
`results/genjl_states/` as portable PPM images.  In the model, each moving
object samples x/y acceleration, clamps velocity to `MAX_SPEED`, and clamps
its position so it stays inside the scene.  Stationary objects are copied
unchanged.
