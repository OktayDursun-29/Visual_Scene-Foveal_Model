using PyCall

# Add this script's folder to Python's import path, independent of the
# directory from which Julia was launched.
pushfirst!(pyimport("sys")."path", joinpath(@__DIR__, "..", "scripts"))

# Import Python modules
rfjax = pyimport("ReceptiveFieldsJAX")

println("Imported modules")


objects = [100.0 100.0 20.0 0.0 255.0 0.0 0.0]
background = [255.0, 255.0, 255.0]
rfs = [100.0 100.0 20.0]


println("Calling JAX function...")


means, variances = pycall(
    rfjax[:predict_rf_stats_arrays_raw],
    Tuple{Matrix{Float64}, Matrix{Float64}},
    objects,
    background,
    rfs,
)


println("Returned:")
println("means = ", means)
println("variances = ", variances)
