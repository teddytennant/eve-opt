EvE (Evolutionary Explorer) is a steady-state differential evolution optimizer with a targeted Adam fallback, following Algorithm 1 of arXiv 2609.35614.

Install from this directory with `pip install -e .`. Run the CPU smoke test with `python -m eve.train --steps 50`. Run the tests with `pytest -q`.

This is a small JAX implementation of the optimizer and a few analytic benchmarks. It does not match the paper's experimental scale: there is no 1e6-dimensional sweep, no MNIST training, no 1.5B-parameter LoRA fine-tune, and no successive halving on UCI Adult Income. Synthetic Adam weight decay defaults to 0 here (Table 4 uses 1e-4). Benchmarks are sphere, ellipsoid, rotated ellipsoid, rosenbrock, rastrigin, ackley, and griewank on an affine image of the unit cube, not the paper's Schwefel and Zakharov suite, and there is no evaluation-cost accounting.
An H200 smoke of `python -m eve.train --steps 50` was submitted as job 748237. The queue start was 2026-10-03 23:17Z, outside this session, so the job was cancelled and the smoke did not run.
