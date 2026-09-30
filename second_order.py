"""Second-order and quasi-Newton methods for smooth unconstrained minimization."""

import numpy as np


def bisect(h, a, b, delta):
    """Find a root of a decreasing function h on [a, b] to within delta."""
    lo, hi = a, b
    while hi - lo > delta:
        m = 0.5 * (lo + hi)
        if h(m) > 0:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


def _trust_region_step(g, H, Delta, delta=1e-10):
    """Solve min_s g.s + 0.5 s.H.s  subject to ||s|| <= Delta.

    Uses an eigendecomposition of H. If H is positive definite and the Newton
    step fits in the region, take it. Otherwise find lambda >= max(0, -lambda_min)
    with ||(H + lambda I)^{-1} g|| = Delta by bisection.
    """
    vals, vecs = np.linalg.eigh(H)
    gt = vecs.T @ g

    def step(lam):
        return -vecs @ (gt / (vals + lam))

    if vals[0] > 0:
        s = step(0.0)
        if np.linalg.norm(s) <= Delta:
            return s

    lam_lo = max(0.0, -vals[0]) + 1e-12
    # ||s(lam)|| is decreasing in lam; find an upper bracket
    lam_hi = lam_lo + 1.0
    while np.linalg.norm(step(lam_hi)) > Delta:
        lam_hi *= 2.0
    lam = bisect(lambda l: np.linalg.norm(step(l)) - Delta, lam_lo, lam_hi, delta)
    return step(lam)


def trust_region_newton(f, grad, hess, x0, Delta0=1.0, T=200, tol=1e-8,
                        eta=0.1, Delta_max=100.0):
    """Trust-region Newton method.

    Returns the iterate trajectory and the squared gradient norm at each iterate.
    """
    x = np.array(x0, dtype=float)
    Delta = float(Delta0)
    traj = [x.copy()]
    grad_norms = [np.linalg.norm(grad(x)) ** 2]

    for _ in range(T):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break
        H = hess(x)
        s = _trust_region_step(g, H, Delta)

        predicted = -(g @ s + 0.5 * s @ (H @ s))
        actual = f(x) - f(x + s)
        rho = actual / predicted if predicted > 0 else 0.0

        if rho < 0.25:
            Delta *= 0.25
        elif rho > 0.75 and np.isclose(np.linalg.norm(s), Delta):
            Delta = min(2 * Delta, Delta_max)

        if rho > eta:
            x = x + s

        traj.append(x.copy())
        grad_norms.append(np.linalg.norm(grad(x)) ** 2)

    return np.array(traj), np.array(grad_norms)


def weak_wolfe(f, grad, x, p, c1=1e-4, c2=0.9, alpha0=1.0, maxiter=60):
    """Bracketing line search for a step satisfying the weak Wolfe conditions."""
    phi0 = f(x)
    der0 = grad(x) @ p
    if der0 >= 0:
        raise ValueError("p is not a descent direction")

    lo, hi = 0.0, np.inf
    alpha = alpha0
    for _ in range(maxiter):
        if f(x + alpha * p) > phi0 + c1 * alpha * der0:
            hi = alpha                      # step too long: shrink
        elif grad(x + alpha * p) @ p < c2 * der0:
            lo = alpha                      # step too short: grow
        else:
            return alpha
        alpha = 2 * lo if np.isinf(hi) else 0.5 * (lo + hi)
    return alpha


def bfgs(f, grad, x0, c1=1e-4, c2=0.9, T=200, tol=1e-8, alpha0=1.0):
    """BFGS with a weak Wolfe line search and inverse-Hessian updates."""
    x = np.array(x0, dtype=float)
    n = len(x)
    I = np.eye(n)
    Hk = I.copy()
    traj = [x.copy()]
    grad_norms = [np.linalg.norm(grad(x)) ** 2]

    for _ in range(T):
        gk = grad(x)
        if np.linalg.norm(gk) < tol:
            break
        pk = -Hk @ gk
        alpha = weak_wolfe(f, grad, x, pk, c1, c2, alpha0)
        sk = alpha * pk
        x_new = x + sk
        yk = grad(x_new) - gk

        sy = sk @ yk
        if sy > 1e-12:   # the Wolfe conditions guarantee this in exact arithmetic
            rho = 1.0 / sy
            Hk = (I - rho * np.outer(sk, yk)) @ Hk @ (I - rho * np.outer(yk, sk)) \
                + rho * np.outer(sk, sk)

        x = x_new
        traj.append(x.copy())
        grad_norms.append(np.linalg.norm(grad(x)) ** 2)

    return np.array(traj), np.array(grad_norms)
