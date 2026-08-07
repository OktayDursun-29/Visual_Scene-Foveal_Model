import jax
import jax.numpy as jnp
import genjax
from genjax import beta, flip, gen, Target, ChoiceMap
from genjax.inference.smc import ImportanceK

# Create a generative model.
@gen
def beta_bernoulli(α, β):
    p = beta(α, β) @ "p"
    v = flip(p) @ "v"
    return v

@jax.jit
def run_inference(obs: bool):
    # Create an inference query - a posterior target - by specifying
    # the model, arguments to the model, and constraints.
    posterior_target = Target(beta_bernoulli, # the model
                              (2.0, 2.0), # arguments to the model
                              ChoiceMap.d({"v": obs}), # constraints
                            )

    # Use a library algorithm, or design your own - more on that in the docs!
    alg = ImportanceK(posterior_target, k_particles=50)

    # Everything is JAX compatible by default.
    # JIT, vmap, to your heart's content.
    key = jax.random.key(314159)
    sub_keys = jax.random.split(key, 50)
    _, p_chm = jax.vmap(alg.random_weighted, in_axes=(0, None))(
        sub_keys, posterior_target
    )

    # An estimate of `p` over 50 independent trials of SIR (with K = 50 particles).
    return jnp.mean(p_chm["p"])

(run_inference(True), run_inference(False))