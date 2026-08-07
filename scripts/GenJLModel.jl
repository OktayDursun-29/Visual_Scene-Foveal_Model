#!/usr/bin/env julia
"""A Gen.jl scene dynamics model using JAX receptive-field statistics via PyCall."""

using Gen
using PyCall
using Random

const IMAGE_WIDTH = 800.0
const IMAGE_HEIGHT = 800.0
const MAX_SPEED = 20.0
const ACCEL_STD = 1.0

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

function simulate_trajectory(initial_scene::SceneState, rfs::Matrix{Float64}; steps::Int=5, seed::Int=0)
    Random.seed!(seed)
    scenes = SceneState[initial_scene]
    observations = Matrix{Float64}[]
    scene = initial_scene
    for _ in 1:steps
        trace = simulate(time_step, (scene, rfs))
        scene, observation = get_retval(trace)
        push!(scenes, scene)
        push!(observations, observation)
    end
    return scenes, observations
end

function render_ppm(scene::SceneState, path::AbstractString; width::Int=800, height::Int=800)
    pixels = fill(scene.background, height, width)
    for obj in scene.objects
        for y in max(1, floor(Int, obj.y - obj.size)):min(height, ceil(Int, obj.y + obj.size)),
            x in max(1, floor(Int, obj.x - obj.size)):min(width, ceil(Int, obj.x + obj.size))
            dx, dy = x - obj.x, y - obj.y
            inside = obj.shape_type == 0 ? dx^2 + dy^2 <= obj.size^2 :
                     obj.shape_type == 1 ? abs(dx) <= obj.size / 2 && abs(dy) <= obj.size / 2 :
                     dy >= -obj.size / 2 && dy <= obj.size / 2 && abs(dx) <= (obj.size / 2 - dy) / 2
            inside && (pixels[y, x] = obj.color)
        end
    end
    open(path, "w") do io
        println(io, "P3\n$width $height\n255")
        for pixel in pixels
            println(io, join(round.(Int, pixel), " "))
        end
    end
end

"""Write five sampled scene states as a static, side-by-side SVG."""
function render_state_grid_svg(scenes::Vector{SceneState}, path::AbstractString)
    panel_width = 260
    panel_height = 300
    scale = 0.3
    open(path, "w") do io
        println(io, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"$(panel_width * length(scenes))\" height=\"$panel_height\">")
        for (index, scene) in enumerate(scenes)
            x_offset = (index - 1) * panel_width
            println(io, "  <g transform=\"translate($x_offset,0)\">")
            println(io, "    <text x=\"10\" y=\"24\" font-family=\"sans-serif\" font-size=\"18\">State $index</text>")
            println(io, "    <g transform=\"translate(10,40) scale($scale)\">")
            println(io, "      <rect width=\"800\" height=\"800\" fill=\"white\" stroke=\"black\" stroke-width=\"4\"/>")
            for object in scene.objects
                color = "rgb($(round(Int, object.color[1])), $(round(Int, object.color[2])), $(round(Int, object.color[3])))"
                if object.shape_type == 0
                    println(io, "      <circle cx=\"$(object.x)\" cy=\"$(object.y)\" r=\"$(object.size)\" fill=\"$color\"/>")
                else
                    half = object.size / 2
                    println(io, "      <rect x=\"$(object.x - half)\" y=\"$(object.y - half)\" width=\"$(object.size)\" height=\"$(object.size)\" fill=\"$color\"/>")
                end
            end
            println(io, "    </g>")
            println(io, "  </g>")
        end
        println(io, "</svg>")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    initial = SceneState([
        SceneObject(300.0, 400.0, 30.0, 0, (255.0, 0.0, 0.0), 8.0, 4.0, false),
        SceneObject(500.0, 400.0, 30.0, 0, (0.0, 0.0, 255.0), 0.0, 0.0, true),
    ], (255.0, 255.0, 255.0))
    rfs = [300.0 400.0 40.0; 500.0 400.0 40.0]
    scenes, observations = simulate_trajectory(initial, rfs; steps=5)
    mkpath(joinpath(@__DIR__, "..", "results", "genjl_states"))
    sampled_states = scenes[2:end]
    for (t, scene) in enumerate(sampled_states)
        render_ppm(scene, joinpath(@__DIR__, "..", "results", "genjl_states", "state_$t.ppm"))
    end
    trajectory_path = joinpath(@__DIR__, "..", "results", "genjl_states", "trajectory_states.svg")
    render_state_grid_svg(sampled_states, trajectory_path)
    println("Sampled $(length(scenes) - 1) time steps.")
    println("Open results/genjl_states/trajectory_states.svg to view the five sampled states.")
end
