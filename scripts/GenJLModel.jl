#!/usr/bin/env julia
"""A Gen.jl scene dynamics model using JAX receptive-field statistics via PyCall."""

using Gen
using PyCall
using Random

const IMAGE_WIDTH = 800.0
const IMAGE_HEIGHT = 800.0
const MAX_SPEED = 20.0
const ACCEL_STD = 1.0
const NUM_FIXATION_PARTICLES = 32
const FIXATION_PREDICTION_HORIZON = 3
const DEFAULT_INITIAL_FIXATION = (IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2)

struct SceneObject
    x::Float64
    y::Float64
    size::Float64
    shape_type::Int       # 0=circle, 1=square, 2=triangle
    color::NTuple{3,Float64}
    vx::Float64
    vy::Float64
    stationary::Bool
end

struct SceneState
    objects::Vector{SceneObject}
    background::NTuple{3,Float64}
end

object_extent(obj::SceneObject) = obj.shape_type == 0 ? obj.size : obj.size / 2
clamp_position(value, extent, upper) = clamp(value, extent, upper - extent)

function as_jax_arrays(scene::SceneState, rfs::Matrix{Float64})
    objects = Matrix{Float64}(undef, length(scene.objects), 7)
    for (i, obj) in enumerate(scene.objects)
        objects[i, :] = [obj.x, obj.y, obj.size, obj.shape_type, obj.color...]
    end
    return objects, collect(scene.background), rfs
end

const rf_jax = let
    pushfirst!(pyimport("sys")."path", @__DIR__)
    pyimport("ReceptiveFieldsJAX")
end

const fixation_optimizer = let
    pushfirst!(pyimport("sys")."path", @__DIR__)
    pyimport("foveal_scene.fixation")
end

function receptive_field_stats(scene::SceneState, rfs::Matrix{Float64})
    objects, background, rfs = as_jax_arrays(scene, rfs)
    pycall(
        rf_jax[:predict_rf_stats_arrays_raw],
        Tuple{Matrix{Float64}, Matrix{Float64}},
        objects,
        background,
        rfs,
    )
end

@gen function motion_prior(scene::SceneState)
    next_objects = SceneObject[]
    for (i, obj) in enumerate(scene.objects)
        if obj.stationary
            push!(next_objects, obj)
            continue
        end
        ax = @trace(normal(0.0, ACCEL_STD), (:ax, i))
        ay = @trace(normal(0.0, ACCEL_STD), (:ay, i))
        vx = clamp(obj.vx + ax, -MAX_SPEED, MAX_SPEED)
        vy = clamp(obj.vy + ay, -MAX_SPEED, MAX_SPEED)
        extent = object_extent(obj)
        x = clamp_position(obj.x + vx, extent, IMAGE_WIDTH)
        y = clamp_position(obj.y + vy, extent, IMAGE_HEIGHT)
        push!(next_objects, SceneObject(x, y, obj.size, obj.shape_type, obj.color, vx, vy, false))
    end
    return SceneState(next_objects, scene.background)
end

@gen function time_step(prev_scene::SceneState, rfs::Matrix{Float64})
    new_scene = @trace(motion_prior(prev_scene), :motion)
    means, variances = receptive_field_stats(new_scene, rfs)
    observed = Matrix{Float64}(undef, size(rfs, 1), 3)
    for rf in axes(rfs, 1), channel in 1:3
        observed[rf, channel] = @trace(
            normal(means[rf, channel], sqrt(max(variances[rf, channel], 1e-6))),
            (:observed, rf, channel),
        )
    end
    return new_scene, observed
end

"""Draw K future motion-prior trajectories as target-position particles."""
function target_position_particles(
    scene::SceneState;
    particles::Int=NUM_FIXATION_PARTICLES,
    horizon::Int=FIXATION_PREDICTION_HORIZON,
)
    n_objects = length(scene.objects)
    samples = Array{Float64, 4}(undef, n_objects, particles, horizon + 1, 2)
    for particle in 1:particles
        predicted_scene = scene
        for (index, object) in enumerate(predicted_scene.objects)
            samples[index, particle, 1, :] = [object.x, object.y]
        end
        for future_time in 1:horizon
            predicted_scene = get_retval(simulate(motion_prior, (predicted_scene,)))
            for (index, object) in enumerate(predicted_scene.objects)
                samples[index, particle, future_time + 1, :] = [object.x, object.y]
            end
        end
    end
    return samples
end

"""Move an RF layout, expressed relative to its initial fixation, to a new fixation."""
function recenter_rfs(
    rf_template::Matrix{Float64},
    reference_fixation::Vector{Float64},
    fixation::Vector{Float64},
)
    rfs = copy(rf_template)
    for index in axes(rfs, 1)
        rfs[index, 1] = clamp(
            rf_template[index, 1] - reference_fixation[1] + fixation[1],
            0.0,
            IMAGE_WIDTH,
        )
        rfs[index, 2] = clamp(
            rf_template[index, 2] - reference_fixation[2] + fixation[2],
            0.0,
            IMAGE_HEIGHT,
        )
    end
    return rfs
end

"""Call the JAX optimizer through PyCall using absolute 800 x 800 image coordinates."""
function resolve_next_fixation(
    fixation::Vector{Float64},
    fixation_velocity::Vector{Float64},
    target_samples::Array{Float64, 4},
    task_relevance::Vector{Float64},
)
    return pycall(
        fixation_optimizer[:resolve_next_fixation_numpy],
        Vector{Float64},
        fixation,
        fixation_velocity,
        target_samples,
        task_relevance;
        bounds=[0.0 0.0; IMAGE_WIDTH IMAGE_HEIGHT],
    )
end

"""Read an optional `--focus X Y` starting fixation from the Julia command line."""
function parse_initial_fixation(args::Vector{String})
    if isempty(args)
        return [DEFAULT_INITIAL_FIXATION[1], DEFAULT_INITIAL_FIXATION[2]]
    end
    length(args) == 3 && args[1] == "--focus" || error(
        "Usage: julia scripts/GenJLModel.jl [--focus X Y]"
    )
    fixation = [parse(Float64, args[2]), parse(Float64, args[3])]
    0.0 <= fixation[1] <= IMAGE_WIDTH || error("focus X must be between 0 and $IMAGE_WIDTH")
    0.0 <= fixation[2] <= IMAGE_HEIGHT || error("focus Y must be between 0 and $IMAGE_HEIGHT")
    return fixation
end

function simulate_trajectory(
    initial_scene::SceneState,
    rfs::Matrix{Float64};
    steps::Int=5,
    seed::Int=0,
    initial_fixation::Vector{Float64}=[DEFAULT_INITIAL_FIXATION[1], DEFAULT_INITIAL_FIXATION[2]],
    task_relevance::Union{Nothing, Vector{Float64}}=nothing,
)
    Random.seed!(seed)
    scenes = SceneState[initial_scene]
    observations = Matrix{Float64}[]
    fixations = Vector{Float64}[copy(initial_fixation)]
    scene = initial_scene
    fixation = copy(initial_fixation)
    fixation_velocity = [0.0, 0.0]
    relevance = task_relevance === nothing ? zeros(length(scene.objects)) : task_relevance
    length(relevance) == length(scene.objects) || error("task_relevance needs one value per object")

    for _ in 1:steps
        # Predict future object positions, select fixation, then observe the
        # actual next sampled scene using RFs centered at that fixation.
        targets = target_position_particles(scene)
        next_fixation = resolve_next_fixation(fixation, fixation_velocity, targets, relevance)
        next_rfs = recenter_rfs(rfs, initial_fixation, next_fixation)
        trace = simulate(time_step, (scene, next_rfs))
        scene, observation = get_retval(trace)
        push!(scenes, scene)
        push!(observations, observation)
        fixation_velocity = next_fixation - fixation
        fixation = next_fixation
        push!(fixations, copy(fixation))
    end
    return scenes, observations, fixations
end

"""Write one animated SVG with one moving red and one stationary blue circle."""
function render_trajectory_svg(scenes::Vector{SceneState}, path::AbstractString)
    isempty(scenes) && error("at least one scene is required")
    all(length(scene.objects) == 2 for scene in scenes) || error(
        "trajectory visualization requires exactly two objects"
    )
    red_positions = [(scene.objects[1].x, scene.objects[1].y) for scene in scenes]
    blue = scenes[1].objects[2]
    x_values = join((position[1] for position in red_positions), ";")
    y_values = join((position[2] for position in red_positions), ";")
    path_points = join(("$(position[1]),$(position[2])" for position in red_positions), " ")

    open(path, "w") do io
        println(io, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"800\" height=\"800\" viewBox=\"0 0 800 800\">")
        println(io, "  <rect width=\"800\" height=\"800\" fill=\"white\" stroke=\"black\" stroke-width=\"4\"/>")
        println(io, "  <text x=\"24\" y=\"40\" font-family=\"sans-serif\" font-size=\"24\">Simulation trajectory</text>")
        println(io, "  <polyline points=\"$path_points\" fill=\"none\" stroke=\"rgb(255, 170, 170)\" stroke-width=\"5\"/>")
        println(io, "  <g id=\"sampled-states\" fill=\"rgb(255, 0, 0)\" fill-opacity=\"0.18\">")
        for (x, y) in red_positions
            println(io, "    <circle cx=\"$x\" cy=\"$y\" r=\"$(scenes[1].objects[1].size)\"/>")
        end
        println(io, "  </g>")
        println(io, "  <circle cx=\"$(blue.x)\" cy=\"$(blue.y)\" r=\"$(blue.size)\" fill=\"rgb(0, 0, 255)\"/>")
        println(io, "  <circle cx=\"$(red_positions[1][1])\" cy=\"$(red_positions[1][2])\" r=\"$(scenes[1].objects[1].size)\" fill=\"rgb(255, 0, 0)\">")
        println(io, "    <animate attributeName=\"cx\" values=\"$x_values\" dur=\"8s\" repeatCount=\"indefinite\"/>")
        println(io, "    <animate attributeName=\"cy\" values=\"$y_values\" dur=\"8s\" repeatCount=\"indefinite\"/>")
        println(io, "  </circle>")
        println(io, "</svg>")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    initial = SceneState([
        SceneObject(300.0, 400.0, 30.0, 0, (255.0, 0.0, 0.0), 14.0, 2.5, false),
        SceneObject(500.0, 400.0, 30.0, 0, (0.0, 0.0, 255.0), 0.0, 0.0, true),
    ], (255.0, 255.0, 255.0))
    initial_fixation = parse_initial_fixation(ARGS)
    # The template is centered at (400, 400); shift it to the requested focus.
    rf_template = [300.0 400.0 40.0; 500.0 400.0 40.0]
    rfs = recenter_rfs(
        rf_template,
        [DEFAULT_INITIAL_FIXATION[1], DEFAULT_INITIAL_FIXATION[2]],
        initial_fixation,
    )
    # The red moving object is task-relevant in this synthetic demo.
    scenes, observations, fixations = simulate_trajectory(
        initial,
        rfs;
        steps=18,
        initial_fixation=initial_fixation,
        task_relevance=[5.0, 0.0],
    )
    mkpath(joinpath(@__DIR__, "..", "results", "genjl_states"))
    trajectory_path = joinpath(@__DIR__, "..", "results", "genjl_states", "trajectory_states.svg")
    render_trajectory_svg(scenes, trajectory_path)
    println("Sampled $(length(scenes) - 1) time steps.")
    for (step, fixation) in enumerate(fixations)
        println("Fixation $(step - 1): ($(round(fixation[1]; digits=2)), $(round(fixation[2]; digits=2)))")
    end
    println("Open results/genjl_states/trajectory_states.svg to view the simulation.")
end
