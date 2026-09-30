"""Sparse signal recovery with lasso, solved by ADMM (with an inner FISTA solver)."""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from optlib import admm_lasso

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    m, n, k, gamma, lam = 200, 500, 10, 0.1, 0.1

    A = rng.standard_normal((m, n)) / np.sqrt(m)
    x_true = np.zeros(n)
    x_true[:k] = 1.0
    x_true[k:2 * k] = -1.0
    b = A @ x_true + gamma * rng.standard_normal(m)    # independent noise draw

    x_hat, primal, dual = admm_lasso(A, b, lam=lam, rho=1.0, max_iters=500, tol=1e-6)

    support_true = np.flatnonzero(x_true)
    support_hat = np.flatnonzero(np.abs(x_hat) > 1e-6)
    signs_match = np.array_equal(np.sign(x_hat[support_true]), np.sign(x_true[support_true]))
    print(f"ADMM iterations: {len(primal)}")
    print(f"Relative error: {np.linalg.norm(x_hat - x_true) / np.linalg.norm(x_true):.3f}")
    print(f"Recovered support exactly: {set(support_hat) == set(support_true)} "
          f"({len(support_hat)} nonzeros found, {len(support_true)} true)")
    print(f"Signs on true support correct: {signs_match}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].semilogy(primal, label="primal residual")
    ax[0].semilogy(dual, label="dual residual")
    ax[0].set_xlabel("ADMM iteration")
    ax[0].set_ylabel("Residual norm")
    ax[0].set_title("ADMM convergence")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].axhline(0, color="0.8", lw=0.8)
    ax[1].plot(x_hat, "C1.", ms=4, label="lasso estimate")
    ax[1].plot(support_true, x_true[support_true], "o", mfc="none", mec="k", ms=6,
               label="true nonzeros")
    ax[1].set_xlabel("Coordinate")
    ax[1].set_title("Sparse recovery")
    ax[1].legend()

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "lasso_admm.png")
    plt.savefig(out, dpi=150)
    print("saved", os.path.normpath(out))
