"""Algorithm 1 of arXiv 2609.35614 (EvE).

Control flow follows the paper, not a loose DE-plus-Adam sketch.

Search lives in z in [0, 1]^n. Real coordinates are
x = xl + z * (xu - xl). The objective f is evaluated on x.
Adam steps run in x, then the result is mapped back and clamped.

Each iteration challenges one round-robin slot i. Donors a, b, c are drawn
uniformly without replacement from the other members, so i is never a donor
(Section 3.3: the challenged slot is not used to build the child). The child is
rand-to-pbest/1,

    zchild = clamp(za + Ft (zb - zc) + F (zbest - za), 0, 1)
    Ft = F * (1 + gamma * (u - 0.5)), u ~ U(0, 1)

which is Equation (2) and Algorithm 1 line 7. gamma defaults to 1e-4.

Polynomial mutation (Equation 3), when enabled, mutates exactly k genes of a
copy of zchild (default k = 1, the paper's max(1, round(n * p)) with p = 1/n).
The mutant replaces zchild only if it is strictly better on the same batch.
sigma is the proximity-init scale (Equation 1). It is not a PM step size.

Adam fallback fires only when the (possibly mutated) child does not strictly
beat slot i. The burst starts at x = map(zchild), not at the slot, and uses
that slot's persistent moments (m, v, t). Weight decay is L2 folded into the
gradient before the moment update: g = grad f(x) + lambda_wd * x. After nadam
steps the point is mapped back, re-evaluated, and kept only if it beats the
child. The slot is overwritten only if the winner strictly beats fi.

On a stochastic objective one batch is drawn at the start of the iteration.
The slot refresh, the child, the PM mutant, and every Adam gradient and value
in that iteration all see that same batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


@dataclass
class EveResult:
    """Population state and the per-iteration record used by the tests."""

    zbest: Any
    xbest: Any
    fbest: float
    z_init_best: Any
    x_init_best: Any
    f_init_best: float
    population: Any
    population_init: Any
    fitness: Any
    fbest_trace: list = field(default_factory=list)
    iterations: list = field(default_factory=list)
    call_batches: list = field(default_factory=list)
    adam_step_count: int = 0
    n_accepted: int = 0
    adam_t: list = field(default_factory=list)


def jittered_scale(u, F: float, gamma: float):
    """Paper scale Ft = F * (1 + gamma * (u - 0.5))."""
    u = jnp.asarray(u, dtype=jnp.float64)
    return F * (1.0 + gamma * (u - 0.5))


def de_child(za, zb, zc, zbest, F: float, Ft):
    """Rand-to-pbest/1 child, clamped to [0, 1]. Returns (clamped, raw)."""
    raw = za + Ft * (zb - zc) + F * (zbest - za)
    return jnp.clip(raw, 0.0, 1.0), raw


def polynomial_delta(zd, r, eta: float = 20.0, zl: float = 0.0, zu: float = 1.0):
    """One-gene polynomial mutation, Equation (3).

    zd_new = zd + delta_q * (zu - zl), with delta_q from Deb's formula.
    r <= 0.5 selects the lower branch, matching the paper.
    """
    zd = jnp.asarray(zd, dtype=jnp.float64)
    r = jnp.asarray(r, dtype=jnp.float64)
    span = jnp.asarray(zu - zl, dtype=jnp.float64)
    zd = jnp.clip(zd, zl, zu)
    delta1 = (zd - zl) / span
    delta2 = (zu - zd) / span
    power = eta + 1.0
    inv = 1.0 / power
    lower = (2.0 * r + (1.0 - 2.0 * r) * (1.0 - delta1) ** power) ** inv - 1.0
    upper = 1.0 - (2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (1.0 - delta2) ** power) ** inv
    delta_q = jnp.where(r <= 0.5, lower, upper)
    updated = zd + delta_q * span
    return jnp.clip(updated, zl, zu)


def sample_mutation_genes(key, n: int, k: int):
    """Draw k distinct gene indices and one U(0, 1) radius each."""
    k = int(max(1, min(int(k), n)))
    key_idx, key_r = jax.random.split(key)
    indices = jax.random.choice(key_idx, n, shape=(k,), replace=False)
    radii = jax.random.uniform(key_r, shape=(k,))
    return indices, radii


def apply_polynomial_mutation(z, indices, radii, eta: float = 20.0):
    """Mutate the listed genes of z. Other coordinates are unchanged."""
    z = jnp.asarray(z, dtype=jnp.float64)
    indices = jnp.asarray(indices)
    radii = jnp.asarray(radii, dtype=jnp.float64)
    mutated = polynomial_delta(z[indices], radii, eta=eta)
    return z.at[indices].set(mutated)


def polynomial_mutation(z, key, eta: float = 20.0, k: int = 1):
    """Exactly k-gene polynomial mutation. Returns the mutant only."""
    n = int(jnp.asarray(z).shape[0])
    indices, radii = sample_mutation_genes(key, n, k)
    return apply_polynomial_mutation(z, indices, radii, eta=eta)


def adam_step(x, g, m, v, t: int, lr: float, beta1: float, beta2: float, eps: float):
    """One standard Adam step. t is the count before this step (0 at init)."""
    t = int(t) + 1
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * jnp.square(g)
    mhat = m / (1.0 - beta1**t)
    vhat = v / (1.0 - beta2**t)
    x = x - lr * mhat / (jnp.sqrt(vhat) + eps)
    return x, m, v, t


def _as_bounds(xl, xu, n: int):
    lo = jnp.broadcast_to(jnp.asarray(xl, dtype=jnp.float64), (n,))
    hi = jnp.broadcast_to(jnp.asarray(xu, dtype=jnp.float64), (n,))
    if not bool(jnp.all(hi > lo)):
        raise ValueError("xu must be strictly greater than xl on every coordinate")
    return lo, hi


def _latin_hypercube(key, npop: int, n: int):
    """One point per stratum, strata permuted independently per coordinate."""
    key_u, key_perm = jax.random.split(key)
    unit = jax.random.uniform(key_u, shape=(npop, n))
    keys = jax.random.split(key_perm, n)
    strata = jnp.arange(npop, dtype=jnp.float64)
    cols = []
    for j in range(n):
        perm = jax.random.permutation(keys[j], npop)
        cols.append((strata[perm] + unit[:, j]) / npop)
    return jnp.stack(cols, axis=1)


def _proximity_init(key, npop: int, n: int, sigma: float):
    """Anchor z1 ~ U(0, 1)^n, other members clamp(z1 + sigma * N(0, I), 0, 1)."""
    key_a, key_e = jax.random.split(key)
    anchor = jax.random.uniform(key_a, shape=(n,))
    if npop == 1:
        return anchor[None, :]
    noise = jax.random.normal(key_e, shape=(npop - 1, n))
    rest = jnp.clip(anchor + sigma * noise, 0.0, 1.0)
    return jnp.concatenate([anchor[None, :], rest], axis=0)


def _batch_token(batch):
    if batch is None:
        return None
    if isinstance(batch, bool):
        return id(batch)
    if isinstance(batch, (int, float)):
        return batch
    if hasattr(batch, "shape") and tuple(getattr(batch, "shape", ())) == ():
        try:
            return batch.item()
        except (TypeError, ValueError, AttributeError):
            return id(batch)
    return id(batch)


def _draw_batch(key, batch_fn):
    if batch_fn is None:
        return int(jax.random.randint(key, (), 0, 2**31 - 1))
    return batch_fn(key)


def _call_value(f, x, batch, stochastic: bool, sink: list, kind: str):
    token = _batch_token(batch)
    sink.append((kind, token))
    if stochastic:
        value = f(x, batch)
    else:
        value = f(x)
    value = jnp.asarray(value, dtype=jnp.float64)
    if value.shape != ():
        raise ValueError("f must return a scalar")
    return value


def _call_grad(f, grad_fn, x, batch, stochastic: bool, sink: list):
    token = _batch_token(batch)
    sink.append(("adam_grad", token))
    if grad_fn is not None:
        grad = grad_fn(x, batch) if stochastic else grad_fn(x)
    elif stochastic:
        grad = jax.grad(lambda xx: jnp.asarray(f(xx, batch), dtype=jnp.float64))(x)
    else:
        grad = jax.grad(lambda xx: jnp.asarray(f(xx), dtype=jnp.float64))(x)
    return jnp.asarray(grad, dtype=jnp.float64)


def _sample_donors(key, npop: int, target: int):
    others = [j for j in range(npop) if j != target]
    if len(others) < 3:
        raise ValueError("npop must be at least 4 so three donors exclude the slot")
    perm = jax.random.permutation(key, len(others))
    picked = [others[int(perm[t])] for t in range(3)]
    return picked[0], picked[1], picked[2]


def minimize(
    f: Callable,
    n: int,
    *,
    iterations: int = 40,
    seed: int = 0,
    npop: int = 4,
    F: float = 0.5,
    gamma: float = 1e-4,
    sigma: float = 0.1,
    eta: float = 20.0,
    p: float | None = None,
    k: int | None = None,
    nadam: int = 7,
    lr: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    lambda_wd: float = 0.0,
    xl: float | Any = 0.0,
    xu: float | Any = 1.0,
    use_pm: bool = True,
    init: str = "lhs",
    stochastic: bool = False,
    batch_fn: Callable | None = None,
    grad_fn: Callable | None = None,
    z_init: Any = None,
) -> EveResult:
    """Run EvE and return the best population member.

    f(x) for a deterministic objective, or f(x, batch) when stochastic=True.
    x is the real vector xl + z * (xu - xl). Default bounds are the unit cube,
    so x = z and benchmarks may map the cube themselves.

    init is "lhs" (synthetic default, Appendix B) or "proximity" (Equation 1,
    neural default). z_init, if given, replaces random initialization and must
    have shape (npop, n) with entries in [0, 1].

    grad_fn, if given, replaces jax.grad for Adam steps. It must read the same
    batch argument the value calls use. A custom grad_fn is not traced through
    f, so it must apply the same batch itself.
    """
    if npop < 4:
        raise ValueError("npop must be at least 4")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if nadam < 0:
        raise ValueError("nadam must be non-negative")
    if init not in ("lhs", "proximity"):
        raise ValueError("init must be 'lhs' or 'proximity'")

    n = int(n)
    lo, hi = _as_bounds(xl, xu, n)
    scale = hi - lo

    def to_x(z):
        return lo + z * scale

    def to_z(x):
        return (x - lo) / scale

    if p is None:
        p = 1.0 / n
    if use_pm:
        # Paper: exactly k = max(1, round(n * p)) genes, which is 1 at p = 1/n.
        if k is None:
            k_eff = max(1, int(round(n * float(p))))
        else:
            k_eff = int(k)
        k_eff = max(1, min(k_eff, n))
    else:
        k_eff = 0

    key = jax.random.PRNGKey(int(seed))
    if z_init is None:
        key, key_init = jax.random.split(key)
        if init == "lhs":
            population = _latin_hypercube(key_init, npop, n)
        else:
            population = _proximity_init(key_init, npop, n, sigma)
    else:
        population = jnp.asarray(z_init, dtype=jnp.float64)
        if population.shape != (npop, n):
            raise ValueError(
                f"z_init must have shape {(npop, n)}, got {tuple(population.shape)}"
            )
        if bool(jnp.any(population < 0.0) or jnp.any(population > 1.0)):
            raise ValueError("z_init must lie in [0, 1]")

    call_batches: list = []
    init_batch = None
    if stochastic:
        key, key_b = jax.random.split(key)
        init_batch = _draw_batch(key_b, batch_fn)

    fitness_rows = []
    for i in range(npop):
        fitness_rows.append(
            _call_value(
                f, to_x(population[i]), init_batch, stochastic, call_batches, "init"
            )
        )
    fitness = jnp.stack(fitness_rows)
    population_init = population
    init_best_i = int(jnp.argmin(fitness))
    z_init_best = population[init_best_i]
    x_init_best = to_x(z_init_best)
    f_init_best = float(fitness[init_best_i])

    states = [
        (jnp.zeros(n, dtype=jnp.float64), jnp.zeros(n, dtype=jnp.float64), 0)
        for _ in range(npop)
    ]

    fbest_trace = [f_init_best]
    records = []
    adam_step_count = 0
    n_accepted = 0

    for t in range(iterations):
        slot = t % npop
        key, key_b, key_d, key_u, key_pm = jax.random.split(key, 5)
        batch = None
        kinds_here: list[str] = []
        tokens_here: list = []

        if stochastic:
            batch = _draw_batch(key_b, batch_fn)
            refreshed = _call_value(
                f, to_x(population[slot]), batch, True, call_batches, "slot"
            )
            kinds_here.append("slot")
            tokens_here.append(_batch_token(batch))
            fitness = fitness.at[slot].set(refreshed)

        f_slot = float(fitness[slot])
        z_slot_before = population[slot]

        a, b, c = _sample_donors(key_d, npop, slot)
        u = float(jax.random.uniform(key_u))
        Ft = float(jittered_scale(u, F, gamma))
        best_i = int(jnp.argmin(fitness))
        zbest_vec = population[best_i]
        z_child, z_raw = de_child(
            population[a], population[b], population[c], zbest_vec, F, Ft
        )
        f_child = _call_value(
            f, to_x(z_child), batch, stochastic, call_batches, "child"
        )
        kinds_here.append("child")
        tokens_here.append(_batch_token(batch))
        f_child_pre_pm = float(f_child)
        pm_accepted = False
        if use_pm:
            z_mut = polynomial_mutation(z_child, key_pm, eta=eta, k=k_eff)
            f_mut = _call_value(
                f, to_x(z_mut), batch, stochastic, call_batches, "pm"
            )
            kinds_here.append("pm")
            tokens_here.append(_batch_token(batch))
            if float(f_mut) < float(f_child):
                z_child = z_mut
                f_child = f_mut
                pm_accepted = True

        f_child_v = float(f_child)
        adam_steps = 0
        f_adam = None
        z_adam = None
        adam_start = None
        if f_child_v < f_slot:
            z_cand = z_child
            f_cand = f_child_v
        else:
            x = to_x(z_child)
            adam_start = z_child
            m, v, step_t = states[slot]
            for _ in range(nadam):
                g = _call_grad(f, grad_fn, x, batch, stochastic, call_batches)
                kinds_here.append("adam_grad")
                tokens_here.append(_batch_token(batch))
                g = g + lambda_wd * x
                x, m, v, step_t = adam_step(
                    x, g, m, v, step_t, lr, beta1, beta2, eps
                )
                x = jnp.clip(x, lo, hi)
                adam_steps += 1
            states[slot] = (m, v, step_t)
            adam_step_count += adam_steps
            z_adam = jnp.clip(to_z(x), 0.0, 1.0)
            f_adam_raw = _call_value(
                f, to_x(z_adam), batch, stochastic, call_batches, "adam_value"
            )
            kinds_here.append("adam_value")
            tokens_here.append(_batch_token(batch))
            f_adam = float(f_adam_raw)
            if f_adam < f_child_v:
                z_cand = z_adam
                f_cand = f_adam
            else:
                z_cand = z_child
                f_cand = f_child_v

        accepted = bool(f_cand < f_slot)
        if accepted:
            population = population.at[slot].set(z_cand)
            fitness = fitness.at[slot].set(jnp.asarray(f_cand, dtype=jnp.float64))
            n_accepted += 1

        fbest_trace.append(float(jnp.min(fitness)))
        records.append(
            {
                "slot": slot,
                "donors": (a, b, c),
                "u": u,
                "Ft": Ft,
                "f_slot": f_slot,
                "f_child_pre_pm": f_child_pre_pm,
                "f_child": f_child_v,
                "f_adam": f_adam,
                "f_cand": float(f_cand),
                "adam_steps": adam_steps,
                "accepted": accepted,
                "pm_accepted": pm_accepted,
                "z_child": z_child,
                "z_raw": z_raw,
                "z_adam": z_adam,
                "adam_start": adam_start,
                "z_cand": z_cand,
                "z_slot_before": z_slot_before,
                "z_slot_after": population[slot],
                "batch": _batch_token(batch),
                "eval_kinds": kinds_here,
                "batch_tokens": tokens_here,
            }
        )

    best_i = int(jnp.argmin(fitness))
    zbest = population[best_i]
    return EveResult(
        zbest=zbest,
        xbest=to_x(zbest),
        fbest=float(fitness[best_i]),
        z_init_best=z_init_best,
        x_init_best=x_init_best,
        f_init_best=f_init_best,
        population=population,
        population_init=population_init,
        fitness=fitness,
        fbest_trace=fbest_trace,
        iterations=records,
        call_batches=call_batches,
        adam_step_count=adam_step_count,
        n_accepted=n_accepted,
        adam_t=[int(state[2]) for state in states],
    )
