# Numerical Optimization from Scratch

Implementations of classical optimization algorithms in Python and NumPy, written without calling any optimization library. Each method comes with a demo that reproduces its convergence plots.

| Method | File | Demo |
|---|---|---|
| Trust-region Newton (exact subproblem via eigendecomposition + bisection) | `optlib/second_order.py` | `examples/himmelblau.py` |
| BFGS with a weak Wolfe line search | `optlib/second_order.py` | `examples/himmelblau.py` |
| Log-barrier interior-point method for linear programs | `optlib/interior_point.py` | `examples/stigler_diet.py` |
| ADMM for lasso, with an inner FISTA solver | `optlib/admm.py` | `examples/lasso_recovery.py` |

## Quick start

```bash
pip install -r requirements.txt
python examples/himmelblau.py
python examples/stigler_diet.py
python examples/lasso_recovery.py
```

Figures are written to `figures/`.

## Results

### Trust-region Newton vs. BFGS on Himmelblau's function

Himmelblau's function has four local minima. Starting from (10, 12.5), both methods converge to machine precision: trust-region Newton in 11 iterations to (3, 2), and BFGS in 15 iterations to a different minimum at (3.584, −1.848). Trust-region Newton shows quadratic convergence near the solution, while BFGS converges superlinearly.

![Himmelblau](figures/himmelblau.png)

### Stigler's diet problem (interior-point method)

The classic 1945 diet problem: find the cheapest combination of foods that meets nine daily nutrient requirements. This uses a 46-food subset of Stigler's data. The barrier method solves a sequence of subproblems with decreasing barrier parameter μ, using damped Newton steps that keep every iterate strictly feasible.

The minimum cost is **$0.1318/day ($48.11/year)**, matching SciPy's `linprog` to machine precision. The optimal diet uses four foods: wheat flour, cheddar cheese, beef liver, and cabbage.

![Stigler](figures/stigler_barrier.png)

### Sparse recovery with lasso (ADMM)

Minimizes ‖Ax − b‖² + λ‖x‖₁ using the splitting y = Ax − b. The x-update is a lasso subproblem solved with FISTA; the y-update has a closed form. The true signal has 20 nonzero entries out of 500, measured through a 200 × 500 Gaussian matrix with noise.

ADMM's primal and dual residuals converge linearly, and the solution matches a long-run reference solve to within 10⁻⁷. At this noise level (γ = 0.1), the estimate captures the true nonzeros with the correct signs, but no choice of λ recovers the support exactly: small λ leaves many false positives, and large λ shrinks true coefficients to zero.

![Lasso](figures/lasso_admm.png)

## Background

These started as assignments in Math 170 (Mathematical Methods for Optimization) at UC Berkeley, Fall 2025, and were rewritten here as a general-purpose library.
