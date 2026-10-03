"""Checks against Algorithm 1 of arXiv 2609.35614, not just a smoke run."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from eve.benchmarks import (
    ackley,
    ellipsoid,
    griewank,
    rastrigin,
    rosenbrock,
    rotated_ellipsoid,
    rotation,
    sphere,
)
from eve.optimizer import (
    apply_polynomial_mutation,
    de_child,
    jittered_scale,
    minimize,
    polynomial_delta,
    sample_mutation_genes,
)

jax.config.update("jax_enable_x64", True)


def _quadratic(z):
    return jnp.sum((z - 0.5) ** 2)


def test_fbest_monotone_on_sphere():
    result = minimize(sphere, n=8, iterations=40, seed=0, use_pm=True, lambda_wd=0.0)
    trace = result.fbest_trace
    assert len(trace) == 41
    for prev, nxt in zip(trace, trace[1:]):
        assert nxt <= prev + 1e-8


def test_adam_skipped_when_child_improves():
    # Slot 0 is the first round-robin target and is the only bad member.
    # Donors are the other three, all at the optimum, so the child is the
    # optimum and strictly beats the slot. Adam must not run.
    n = 4
    z0 = jnp.ones((4, n)) * 0.5
    z0 = z0.at[0].set(jnp.full(n, 0.9))
    result = minimize(
        _quadratic,
        n=n,
        iterations=1,
        seed=0,
        z_init=z0,
        use_pm=True,
        nadam=7,
        lambda_wd=0.0,
    )
    rec = result.iterations[0]
    assert rec["slot"] == 0
    assert rec["f_child"] < rec["f_slot"]
    assert rec["adam_steps"] == 0
    assert rec["f_adam"] is None
    assert result.adam_step_count == 0
    assert rec["accepted"]
    assert rec["f_cand"] < rec["f_slot"]
    # PM accept/reject cannot make the child worse.
    assert rec["f_child"] <= rec["f_child_pre_pm"] + 1e-12


def test_adam_invoked_when_child_does_not_improve():
    # Identical members at the optimum: the child equals the slot, so the
    # strict test fails and the Adam burst runs. Zero gradient and zero weight
    # decay leave the point unimproved, so the slot is not overwritten.
    n = 3
    z0 = jnp.full((4, n), 0.5)
    result = minimize(
        _quadratic,
        n=n,
        iterations=2,
        seed=1,
        z_init=z0,
        use_pm=False,
        nadam=4,
        lambda_wd=0.0,
        lr=0.05,
    )
    for rec in result.iterations:
        assert rec["f_child"] >= rec["f_slot"] - 1e-12
        assert rec["adam_steps"] == 4
        assert not rec["accepted"]
        assert jnp.allclose(rec["z_slot_after"], rec["z_slot_before"])
    assert result.adam_step_count == 8
    assert jnp.allclose(result.population, z0)


def test_adam_starts_from_child_not_slot():
    # f(z) = sum(z). Slot 0 is at 0 (best). Donors are at 1, so
    # zchild = 1 + F * (0 - 1) = 0.5, which is worse than the slot.
    # One Adam step from 0.5 with a unit gradient is a known closed form.
    n = 2
    z0 = jnp.ones((4, n))
    z0 = z0.at[0].set(jnp.zeros(n))

    def f(z):
        return jnp.sum(z)

    lr = 0.05
    eps = 1e-8
    result = minimize(
        f,
        n=n,
        iterations=1,
        seed=0,
        z_init=z0,
        use_pm=False,
        nadam=1,
        lr=lr,
        lambda_wd=0.0,
        beta1=0.9,
        beta2=0.999,
        eps=eps,
    )
    rec = result.iterations[0]
    assert rec["adam_steps"] == 1
    assert len(set(rec["donors"])) == 3
    assert 0 not in rec["donors"]
    expected_child = 1.0 - 0.5  # F * (zbest - za) with za = 1, zbest = 0
    assert jnp.allclose(rec["adam_start"], expected_child)
    # First Adam step, moments were 0: mhat = g, vhat = g^2, g = 1.
    expected = 0.5 - lr * 1.0 / (1.0 + eps)
    assert jnp.allclose(rec["z_adam"], expected, atol=1e-12)
    # Slot was already optimal, so the Adam point is not accepted.
    assert not rec["accepted"]
    assert jnp.allclose(result.population[0], 0.0)
    # Moments persist on the challenged slot even though the point was rejected.
    assert result.adam_t[0] == 1
    assert result.adam_t[1] == 0


def test_slot_replaced_only_on_strict_improvement():
    result = minimize(sphere, n=8, iterations=40, seed=2, use_pm=True, lambda_wd=0.0)
    assert result.iterations
    saw_accept = False
    saw_reject = False
    for rec in result.iterations:
        if rec["accepted"]:
            saw_accept = True
            assert rec["f_cand"] < rec["f_slot"]
            assert jnp.allclose(rec["z_slot_after"], rec["z_cand"])
        else:
            saw_reject = True
            assert rec["f_cand"] >= rec["f_slot"] - 1e-12
            assert jnp.allclose(rec["z_slot_after"], rec["z_slot_before"])
        # Round-robin target, donors distinct and not the target (Algorithm 1).
        assert len(set(rec["donors"])) == 3
        assert rec["slot"] not in rec["donors"]
    assert saw_accept
    assert saw_reject
    slots = [rec["slot"] for rec in result.iterations]
    assert slots == [t % 4 for t in range(40)]


def test_common_random_numbers():
    seen = []

    def f(z, batch):
        seen.append(int(batch))
        return jnp.sum((z - 0.5) ** 2) + 0.001 * int(batch)

    counter = {"n": 0}

    def batch_fn(_key):
        counter["n"] += 1
        return counter["n"]

    z0 = jnp.full((4, 3), 0.5)
    result = minimize(
        f,
        n=3,
        iterations=4,
        seed=0,
        z_init=z0,
        stochastic=True,
        batch_fn=batch_fn,
        use_pm=True,
        nadam=2,
        lambda_wd=0.0,
    )
    # Child equals the slot, so every iteration refreshes, scores the child
    # and the mutant, then runs the Adam burst, all on one batch.
    assert seen == [token for _kind, token in result.call_batches]
    iter_batches = []
    for rec in result.iterations:
        assert "slot" in rec["eval_kinds"]
        assert "child" in rec["eval_kinds"]
        assert "pm" in rec["eval_kinds"]
        assert rec["eval_kinds"].count("adam_grad") == 2
        assert "adam_value" in rec["eval_kinds"]
        assert len(rec["batch_tokens"]) == len(rec["eval_kinds"])
        assert all(token == rec["batch"] for token in rec["batch_tokens"])
        iter_batches.append(rec["batch"])
    assert iter_batches == [2, 3, 4, 5]
    assert len(set(iter_batches)) == 4


def _numpy_delta(zd, r, eta, zl=0.0, zu=1.0):
    """Independent transcription of Equation (3). Not a call into eve."""
    span = zu - zl
    delta1 = (zd - zl) / span
    delta2 = (zu - zd) / span
    power = eta + 1.0
    if r <= 0.5:
        delta_q = (2.0 * r + (1.0 - 2.0 * r) * (1.0 - delta1) ** power) ** (
            1.0 / power
        ) - 1.0
    else:
        delta_q = 1.0 - (
            2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (1.0 - delta2) ** power
        ) ** (1.0 / power)
    return zd + delta_q * span


def test_child_clamped_and_pm_matches_paper():
    # Hand check: eta = 1, z = 0.25, r = 0.
    # inner = (1 - 0.25)^2 = 0.5625, sqrt = 0.75, delta_q = -0.25, z_new = 0.
    hand = float(polynomial_delta(0.25, 0.0, eta=1.0))
    assert hand == pytest.approx(0.0, abs=1e-12)
    assert _numpy_delta(0.25, 0.0, eta=1.0) == pytest.approx(0.0, abs=1e-12)

    rng = np.random.default_rng(0)
    for _ in range(30):
        zd = float(rng.uniform(0.0, 1.0))
        r = float(rng.uniform(0.0, 1.0))
        eta = float(rng.choice([1.0, 5.0, 20.0]))
        got = float(polynomial_delta(zd, r, eta=eta))
        ref = _numpy_delta(zd, r, eta)
        assert got == pytest.approx(ref, abs=1e-12)
        assert 0.0 <= got <= 1.0

    # Boundary genes the formula must not push outside the box.
    assert float(polynomial_delta(0.0, 0.0, eta=20.0)) == pytest.approx(0.0, abs=1e-12)
    assert float(polynomial_delta(1.0, 1.0, eta=20.0)) == pytest.approx(1.0, abs=1e-12)

    z = jnp.array([0.2, 0.4, 0.6, 0.8])
    key = jax.random.PRNGKey(7)
    indices, radii = sample_mutation_genes(key, 4, 1)
    got = np.asarray(apply_polynomial_mutation(z, indices, radii, eta=20.0))
    ref = np.array(z, dtype=np.float64)
    for index, radius in zip(np.asarray(indices), np.asarray(radii)):
        ref[int(index)] = _numpy_delta(float(ref[int(index)]), float(radius), 20.0)
    assert np.allclose(got, ref, atol=1e-12)
    assert np.count_nonzero(np.abs(got - np.asarray(z)) > 1e-15) <= 1

    # DE child that would leave the cube is clamped, and Ft matches Equation (2).
    u = 0.2
    F = 0.5
    gamma = 1e-4
    Ft = float(jittered_scale(u, F, gamma))
    assert Ft == pytest.approx(F * (1.0 + gamma * (u - 0.5)), abs=1e-15)
    za = jnp.ones(3)
    zb = jnp.ones(3)
    zc = jnp.zeros(3)
    zbest = jnp.ones(3)
    clamped, raw = de_child(za, zb, zc, zbest, F, Ft)
    assert float(raw[0]) > 1.0
    assert jnp.allclose(clamped, 1.0)
    clamped_lo, raw_lo = de_child(jnp.zeros(3), jnp.zeros(3), jnp.ones(3), jnp.zeros(3), F, Ft)
    assert float(raw_lo[0]) < 0.0
    assert jnp.allclose(clamped_lo, 0.0)

    run = minimize(sphere, n=5, iterations=15, seed=3, use_pm=True)
    for rec in run.iterations:
        assert jnp.all(rec["z_child"] >= 0.0)
        assert jnp.all(rec["z_child"] <= 1.0)
        assert math.isfinite(rec["Ft"])


def test_sphere_improves():
    result = minimize(
        sphere,
        n=8,
        iterations=40,
        seed=0,
        use_pm=True,
        nadam=7,
        lr=0.05,
        lambda_wd=0.0,
        init="lhs",
    )
    assert result.fbest < 0.5 * result.f_init_best
    assert result.fbest == pytest.approx(result.fbest_trace[-1], abs=1e-12)


def test_optima_inside_unit_cube():
    n = 6
    origin = jnp.full(n, 0.5)
    assert float(sphere(origin)) < 1e-8
    assert float(ellipsoid(origin)) < 1e-8
    assert float(rotated_ellipsoid(origin)) < 1e-8
    assert float(rastrigin(origin)) < 1e-6
    assert float(ackley(origin)) < 1e-6
    assert float(griewank(origin)) < 1e-6
    z_rosen = jnp.full(n, (1.0 - (-5.0)) / 10.0)
    assert float(rosenbrock(z_rosen)) < 1e-8
    q = np.asarray(rotation(n))
    assert np.allclose(q.T @ q, np.eye(n), atol=1e-8)
    assert abs(np.linalg.det(q) - 1.0) < 1e-6
