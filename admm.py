"""ADMM for the lasso problem

    minimize ||A x - b||_2^2 + lam * ||x||_1

using the splitting y = A x - b. The x-update is 
solved with FISTA.
"""

import numpy as np


def soft_threshold(v, thresh):
    """Proximal operator of thresh * ||.||_1."""
    return np.sign(v) * np.maximum(np.abs(v) - thresh, 0.0)


def spectral_norm(A, iters=50, seed=0):
    """Largest singular value of A by power iteration."""
    v = np.random.default_rng(seed).standard_normal(A.shape[1])
    for _ in range(iters):
        v = A.T @ (A @ v)
        v /= np.linalg.norm(v)
    return np.linalg.norm(A @ v)


def fista(A, d, rho, lam, x0, L, max_iters=200, tol=1e-8):
    """Minimize lam*||x||_1 + (rho/2)||A x - d||^2 with FISTA."""
    x = x0.copy()
    y = x.copy()
    tk = 1.0
    step = 1.0 / L
    for _ in range(max_iters):
        grad = rho * (A.T @ (A @ y - d))
        x_next = soft_threshold(y - step * grad, lam * step)
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * tk**2)) / 2.0
        y = x_next + ((tk - 1.0) / t_next) * (x_next - x)
        converged = np.linalg.norm(x_next - x) < tol
        x, tk = x_next, t_next
        if converged:
            break
    return x


def admm_lasso(A, b, lam, rho=1.0, max_iters=500, tol=1e-6):
    """Returns the estimate and the primal, dual residual norm histories."""
    m, n = A.shape
    L = rho * spectral_norm(A) ** 2
    x = np.zeros(n)
    y = A @ x - b
    u = np.zeros(m)
    primal, dual = [], []

    for _ in range(max_iters):
        x = fista(A, b + y - u, rho, lam, x, L)            # x-update
        Axmb = A @ x - b
        y_old = y
        y = (rho / (2.0 + rho)) * (Axmb + u)               # y-update (closed form)
        u = u + Axmb - y                                   # scaled dual update

        r = np.linalg.norm(Axmb - y)
        s = np.linalg.norm(rho * (A.T @ (y - y_old)))
        primal.append(r)
        dual.append(s)
        if r < tol and s < tol:
            break

    return x, np.array(primal), np.array(dual)
