"""Trust-region Newton vs. BFGS on Himmelblau's function."""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from optlib import trust_region_newton, bfgs


def f(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


def grad(x):
    a = x[0]**2 + x[1] - 11
    b = x[0] + x[1]**2 - 7
    return np.array([4 * x[0] * a + 2 * b, 2 * a + 4 * x[1] * b])


def hess(x):
    a = x[0]**2 + x[1] - 11
    b = x[0] + x[1]**2 - 7
    return np.array([[12 * x[0]**2 + 4 * x[1] - 42, 4 * (x[0] + x[1])],
                     [4 * (x[0] + x[1]), 4 * x[0] + 12 * x[1]**2 - 26]])


if __name__ == "__main__":
    x0 = np.array([10.0, 12.5])
    tr_traj, tr_g = trust_region_newton(f, grad, hess, x0)
    bf_traj, bf_g = bfgs(f, grad, x0)

    for name, traj, g in [("Trust-region Newton", tr_traj, tr_g), ("BFGS", bf_traj, bf_g)]:
        print(f"{name}: {len(traj) - 1} iterations, x* = {np.round(traj[-1], 6)}, "
              f"||grad||^2 = {g[-1]:.2e}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    floor = 1e-30   # exact zeros can't be drawn on a log axis
    ax[0].semilogy(np.maximum(tr_g, floor), "o-", ms=3, label="Trust-region Newton")
    ax[0].semilogy(np.maximum(bf_g, floor), "x-", ms=4, label="BFGS")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel(r"$\|\nabla f(x_t)\|^2$")
    ax[0].set_title("Convergence")
    ax[0].legend()

    xs = np.linspace(-5, 15, 400)
    X, Y = np.meshgrid(xs, xs)
    Z = (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2
    ax[1].contour(X, Y, Z, levels=np.logspace(0, 4.5, 30), cmap="viridis")
    ax[1].plot(tr_traj[:, 0], tr_traj[:, 1], "o-", ms=3, label="Trust-region Newton")
    ax[1].plot(bf_traj[:, 0], bf_traj[:, 1], "x-", ms=4, label="BFGS")
    ax[1].set_xlim(-5, 15)
    ax[1].set_ylim(-5, 15)
    ax[1].set_title("Trajectories on Himmelblau's function")
    ax[1].legend()

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "himmelblau.png")
    plt.savefig(out, dpi=150)
    print("saved", os.path.normpath(out))
