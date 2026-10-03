"""Tiny MLP smoke test for EvE neural settings.

Two hidden layers, synthetic binary classification, no MNIST. Polynomial
mutation is off, the population is proximity-initialized, the Adam fallback
takes nadam=3 steps, and L2 weight decay is 5e-4, matching Section 3 and
Table 7 rather than the synthetic Table 4 settings.

    python -m eve.train --steps 50
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from eve.optimizer import minimize

jax.config.update("jax_enable_x64", True)

# Real-weight box. Adam steps in this box, as Algorithm 1 does in x.
WEIGHT_BOUND = 1.0
HIDDEN = 8
N_SAMPLES = 256
N_FEATURES = 2
BATCH = 32


def _shapes():
    return ((N_FEATURES, HIDDEN), (HIDDEN, HIDDEN), (HIDDEN, 1))


def n_params() -> int:
    return sum(fan_in * fan_out + fan_out for fan_in, fan_out in _shapes())


def unpack(x):
    params = []
    cursor = 0
    flat = jnp.asarray(x, dtype=jnp.float64)
    for fan_in, fan_out in _shapes():
        width = fan_in * fan_out
        weight = flat[cursor : cursor + width].reshape(fan_in, fan_out)
        cursor += width
        bias = flat[cursor : cursor + fan_out]
        cursor += fan_out
        params.append((weight, bias))
    return params


def logits(x, features):
    hidden = jnp.asarray(features, dtype=jnp.float64)
    layers = unpack(x)
    for index, (weight, bias) in enumerate(layers):
        hidden = hidden @ weight + bias
        if index < len(layers) - 1:
            hidden = jax.nn.relu(hidden)
    return hidden[..., 0]


def bce_with_logits(scores, labels):
    scores = jnp.asarray(scores, dtype=jnp.float64)
    labels = jnp.asarray(labels, dtype=jnp.float64)
    return jnp.mean(
        jnp.maximum(scores, 0.0)
        - scores * labels
        + jnp.log1p(jnp.exp(-jnp.abs(scores)))
    )


def make_data(seed: int):
    key = jax.random.PRNGKey(seed)
    key_x, key_y = jax.random.split(key)
    features = jax.random.normal(key_x, (N_SAMPLES, N_FEATURES))
    labels = (features[:, 0] + 0.5 * features[:, 1] > 0.0).astype(jnp.float64)
    # A little label noise so the loss is not a trivial step function in data.
    flip = jax.random.uniform(key_y, (N_SAMPLES,)) < 0.05
    labels = jnp.where(flip, 1.0 - labels, labels)
    return features, labels


def main():
    parser = argparse.ArgumentParser(description="Train a tiny MLP with EvE")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    features, labels = make_data(args.seed)
    dim = n_params()

    @jax.jit
    def batch_loss(weights, index):
        return bce_with_logits(logits(weights, features[index]), labels[index])

    @jax.jit
    def batch_grad(weights, index):
        return jax.grad(batch_loss)(weights, index)

    def objective(weights, batch):
        return batch_loss(weights, batch)

    def batch_fn(key):
        return jax.random.choice(key, N_SAMPLES, shape=(BATCH,), replace=False)

    def full_loss(weights):
        return float(bce_with_logits(logits(weights, features), labels))

    result = minimize(
        objective,
        dim,
        iterations=args.steps,
        seed=args.seed,
        npop=4,
        nadam=3,
        lr=0.02,
        lambda_wd=5e-4,
        xl=-WEIGHT_BOUND,
        xu=WEIGHT_BOUND,
        use_pm=False,
        init="proximity",
        sigma=0.1,
        stochastic=True,
        batch_fn=batch_fn,
        grad_fn=batch_grad,
    )
    start = full_loss(result.x_init_best)
    end = full_loss(result.xbest)
    print(f"steps: {args.steps}")
    print(f"train loss start: {start:.6f}")
    print(f"train loss end: {end:.6f}")


if __name__ == "__main__":
    main()
