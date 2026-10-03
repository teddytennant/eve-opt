"""Analytic benchmarks on the unit cube.

Each function takes z in [0, 1]^n and affinely maps it onto a usual box before
evaluating the textbook formula. The global minimizer of every function below
lies in the open cube, so a correct bounded optimizer can drive f to about 0.

    sphere, ellipsoid, rotated ellipsoid: x = -5 + 10 z, optimum x = 0 (z = 0.5)
    rosenbrock: x = -5 + 10 z, optimum x = 1 (z = 0.6)
    rastrigin: x = -5.12 + 10.24 z, optimum x = 0 (z = 0.5)
    ackley: x = -32.768 + 65.536 z, optimum x = 0 (z = 0.5)
    griewank: x = -100 + 200 z, optimum x = 0 (z = 0.5)

Rotated ellipsoid uses one fixed orthogonal matrix from numpy Generator seed 0
(QR of a standard normal matrix, first column flipped if needed so det Q = 1).
Ellipsoid weights are 10^(6 i / (n-1)), the usual ill-conditioned diagonal.
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Documented boxes (lo, hi) for the real vector x = lo + (hi - lo) * z.
SPHERE_BOX = (-5.0, 5.0)
ELLIPSOID_BOX = (-5.0, 5.0)
ROSENBROCK_BOX = (-5.0, 5.0)
RASTRIGIN_BOX = (-5.12, 5.12)
ACKLEY_BOX = (-32.768, 32.768)
GRIEWANK_BOX = (-100.0, 100.0)

_ROTATION_CACHE: dict[int, np.ndarray] = {}


def affine(z, lo: float, hi: float):
    """Map z in [0, 1] onto [lo, hi]."""
    z = jnp.asarray(z, dtype=jnp.float64)
    return lo + z * (hi - lo)


def ellipsoid_weights(n: int):
    """Diagonal condition weights, 1 at i = 0 and 1e6 at i = n - 1."""
    if n <= 1:
        return jnp.ones((max(n, 0),), dtype=jnp.float64)
    exponents = 6.0 * jnp.arange(n, dtype=jnp.float64) / (n - 1)
    return jnp.power(10.0, exponents)


def rotation(n: int) -> jnp.ndarray:
    """Fixed orthogonal Q in R^{n x n} with det 1, shared by every call."""
    if n not in _ROTATION_CACHE:
        rng = np.random.default_rng(0)
        raw = rng.normal(size=(n, n))
        q, _r = np.linalg.qr(raw)
        q = np.array(q, dtype=np.float64, copy=True)
        if np.linalg.det(q) < 0.0:
            q[:, 0] *= -1.0
        _ROTATION_CACHE[n] = q
    return jnp.asarray(_ROTATION_CACHE[n], dtype=jnp.float64)


def sphere(z):
    x = affine(z, *SPHERE_BOX)
    return jnp.sum(x * x)


def ellipsoid(z):
    x = affine(z, *ELLIPSOID_BOX)
    weights = ellipsoid_weights(int(x.shape[0]))
    return jnp.sum(weights * x * x)


def rotated_ellipsoid(z):
    x = affine(z, *ELLIPSOID_BOX)
    n = int(x.shape[0])
    y = rotation(n) @ x
    weights = ellipsoid_weights(n)
    return jnp.sum(weights * y * y)


def rosenbrock(z):
    x = affine(z, *ROSENBROCK_BOX)
    if x.shape[0] < 2:
        return jnp.asarray(0.0, dtype=jnp.float64)
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def rastrigin(z):
    x = affine(z, *RASTRIGIN_BOX)
    n = x.shape[0]
    return 10.0 * n + jnp.sum(x * x - 10.0 * jnp.cos(2.0 * jnp.pi * x))


def ackley(z):
    x = affine(z, *ACKLEY_BOX)
    mean_sq = jnp.mean(x * x)
    mean_cos = jnp.mean(jnp.cos(2.0 * jnp.pi * x))
    return (
        -20.0 * jnp.exp(-0.2 * jnp.sqrt(mean_sq))
        - jnp.exp(mean_cos)
        + 20.0
        + jnp.e
    )


def griewank(z):
    x = affine(z, *GRIEWANK_BOX)
    coords = jnp.arange(1, x.shape[0] + 1, dtype=jnp.float64)
    return 1.0 + jnp.sum(x * x) / 4000.0 - jnp.prod(jnp.cos(x / jnp.sqrt(coords)))
