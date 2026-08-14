"""JAX optimization of a task-relevant, smooth next fixation location."""

from __future__ import annotations

from jax import grad, jit
import jax
import jax.numpy as jnp


@jit
def foveal_coverage_loss(
    fixation: jnp.ndarray,
    target_samples: jnp.ndarray,
    weights: jnp.ndarray,
    sigma_fovea: float = 50.0,
    gamma: float = 0.9,
) -> jnp.ndarray:
    """Return negative RF-particle-weighted foveal coverage.

    ``target_samples`` has shape ``(N_obj, K_particles, H+1, 2)``.  A
    Gaussian foveal profile converts distance into coverage, and coverage is
    averaged over particle samples before object relevance and temporal
    discounting are applied.
    """
    if target_samples.ndim != 4 or target_samples.shape[-1] != 2:
        raise ValueError("target_samples must have shape (N_obj, K_particles, H+1, 2)")
    if fixation.shape != (2,):
        raise ValueError("fixation must have shape (2,)")
    if weights.ndim != 1 or weights.shape[0] != target_samples.shape[0]:
        raise ValueError("weights must have one entry per object")
    # (N_obj, K_particles, H+1): distance from this candidate gaze location.
    distances = jnp.linalg.norm(target_samples - fixation, axis=-1)
    # Convert distance into [0, 1] coverage, then compute E[coverage].
    coverage = jnp.exp(-0.5 * (distances / sigma_fovea) ** 2)
    expected_coverage = jnp.mean(coverage, axis=1)  # (N_obj, H+1)
    temporal_discount = gamma ** jnp.arange(target_samples.shape[2])  # (H+1,)
    total_coverage = jnp.sum(expected_coverage * weights[:, None] * temporal_discount)
    return -total_coverage


@jit
def movement_cost(
    fixation: jnp.ndarray,
    fixation_prev: jnp.ndarray,
    fixation_vel: jnp.ndarray,
    lambda_l2: float = 0.0001,
    lambda_smooth: float = 0.0005,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """Penalize both saccade distance and deviation from prior velocity."""
    if fixation.shape != (2,) or fixation_prev.shape != (2,) or fixation_vel.shape != (2,):
        raise ValueError("fixation, fixation_prev, and fixation_vel must each have shape (2,)")
    displacement = fixation - fixation_prev
    movement = jnp.sqrt(jnp.sum(displacement**2) + eps)
    acceleration = displacement - fixation_vel
    return lambda_l2 * movement + lambda_smooth * jnp.sum(acceleration**2)


@jit
def total_fixation_loss(
    fixation: jnp.ndarray,
    fixation_prev: jnp.ndarray,
    fixation_vel: jnp.ndarray,
    target_samples: jnp.ndarray,
    weights: jnp.ndarray,
    sigma_fovea: float = 50.0,
    gamma: float = 0.9,
    lambda_l2: float = 0.0001,
    lambda_smooth: float = 0.0005,
) -> jnp.ndarray:
    """Combined task-relevance and movement objective to minimize."""
    return foveal_coverage_loss(fixation, target_samples, weights, sigma_fovea, gamma) + movement_cost(
        fixation, fixation_prev, fixation_vel, lambda_l2, lambda_smooth
    )


@jit
def resolve_next_fixation_sgd(
    f_t: jnp.ndarray,
    v_t: jnp.ndarray,
    target_samples: jnp.ndarray,
    task_relevance: jnp.ndarray,
    eta_saccade: float = 0.05,
    lr: float = 200.0,
    momentum: float = 0.9,
    num_steps: int = 100,
    bounds: jnp.ndarray = jnp.array([[-400.0, -400.0], [400.0, 400.0]]),
    tau_importance: float = 1.0,
    sigma_fovea: float = 50.0,
    gamma: float = 0.9,
    lambda_l2: float = 0.0001,
    lambda_smooth: float = 0.0005,
) -> jnp.ndarray:
    """Optimize and threshold the next fixation using momentum SGD.

    ``f_t`` is the current fixation and ``v_t`` is its previous displacement
    (both in pixels per simulation timestep). ``task_relevance`` contains one
    unnormalized importance value per object and is normalized by softmax.
    """
    if f_t.shape != (2,) or v_t.shape != (2,):
        raise ValueError("f_t and v_t must each have shape (2,)")
    if bounds.shape != (2, 2):
        raise ValueError("bounds must have shape (2, 2)")
    if task_relevance.shape != (target_samples.shape[0],):
        raise ValueError("task_relevance must have one entry per object")
    weights = jax.nn.softmax(task_relevance / tau_importance)

    def loss_fn(fixation: jnp.ndarray) -> jnp.ndarray:
        return total_fixation_loss(
            fixation,
            f_t,
            v_t,
            target_samples,
            weights,
            sigma_fovea,
            gamma,
            lambda_l2,
            lambda_smooth,
        )

    grad_fn = grad(loss_fn)

    def sgd_step(_: int, value: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        fixation, velocity_momentum = value
        gradient = grad_fn(fixation)
        next_momentum = momentum * velocity_momentum + lr * gradient
        next_fixation = jnp.clip(fixation - next_momentum, bounds[0], bounds[1])
        return next_fixation, next_momentum

    f_opt, _ = jax.lax.fori_loop(0, num_steps, sgd_step, (f_t, jnp.zeros(2)))
    gain = loss_fn(f_t) - loss_fn(f_opt)
    # With insufficient improvement, continue smooth pursuit instead of saccading.
    return jnp.where(gain > eta_saccade, f_opt, f_t + v_t)


def resolve_next_fixation_numpy(
    f_t: object,
    v_t: object,
    target_samples: object,
    task_relevance: object,
    *,
    bounds: object | None = None,
) -> list[float]:
    """Python/Julia bridge returning a plain two-element fixation list.

    This wrapper converts ordinary NumPy arrays or Julia arrays to JAX arrays
    and converts the JAX result back to Python scalars, which PyCall can turn
    directly into ``Vector{Float64}``.
    """
    if bounds is None:
        bounds = jnp.array([[-400.0, -400.0], [400.0, 400.0]])
    result = resolve_next_fixation_sgd(
        jnp.asarray(f_t),
        jnp.asarray(v_t),
        jnp.asarray(target_samples),
        jnp.asarray(task_relevance),
        bounds=jnp.asarray(bounds),
    )
    return [float(value) for value in result]
