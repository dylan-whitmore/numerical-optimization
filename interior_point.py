"""Barrier interior-point method for linear programs of the form

    minimize    c.x
    subject to  A x >= b,  x >= 0.
"""

import numpy as np


def _barrier(x, s, c, mu):
    return c @ x - mu * np.sum(np.log(x)) - mu * np.sum(np.log(s))


def interior_point(A, b, c, x0, mus=None, tol=1e-8, max_newton_iters=80, reg=1e-10):
    """Solve a sequence of log-barrier subproblems with damped Newton steps.
    Returns the final iterate and for each barrier parameter mu, the Newton-step gradient norms.
    """
    m, n = A.shape
    x = np.array(x0, dtype=float)
    if np.any(x <= 0) or np.any(A @ x - b <= 0):
        raise ValueError("x0 must be strictly feasible")
    if mus is None:
        mus = [10.0 ** (-k) for k in range(0, 9)]

    history = {}
    for mu in mus:
        grads = []
        for _ in range(max_newton_iters):
            s = A @ x - b
            grad = c - mu / x - mu * (A.T @ (1.0 / s))
            gnorm = np.linalg.norm(grad)
            grads.append(gnorm)
            if gnorm < tol:
                break

            H = np.diag(mu / x**2) + mu * (A.T * (1.0 / s**2)) @ A + reg * np.eye(n)
            p = np.linalg.solve(H, -grad)

            # backtracking line search keeps the iterate strictly feasible
            t, f0 = 1.0, _barrier(x, s, c, mu)
            while t > 1e-16:
                x_new = x + t * p
                s_new = A @ x_new - b
                if np.all(x_new > 0) and np.all(s_new > 0) and \
                        _barrier(x_new, s_new, c, mu) <= f0 + 1e-4 * t * (grad @ p):
                    break
                t *= 0.5
            else:
                break   # no step; move to the next mu
            x = x_new
        history[mu] = np.array(grads)
    return x, history
