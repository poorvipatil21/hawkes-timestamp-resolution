"""Exponential-kernel Hawkes: simulation, likelihood, MLE, timestamp corruptions.
Intensity: lam(t) = mu + n*beta * sum_{t_i<t} exp(-beta (t - t_i)),  n = branching ratio.
Time unit is chosen so that the TRUE beta = 1; resolution delta is therefore in units of
the kernel decay time (delta*beta = delta)."""
import numpy as np
from numba import njit
from scipy.optimize import minimize

@njit(cache=True)
def simulate(mu, n, beta, T, seed):
    np.random.seed(seed)
    out = np.empty(2_000_000)
    k = 0
    t = 0.0
    S = 0.0            # sum_j n*beta*exp(-beta(t-t_j)) at current t
    while True:
        lam_bar = mu + S
        w = -np.log(np.random.random()) / lam_bar
        t_new = t + w
        if t_new > T:
            break
        S *= np.exp(-beta * w)
        t = t_new
        if np.random.random() * lam_bar <= mu + S:
            out[k] = t
            k += 1
            S += n * beta
    return out[:k].copy()

@njit(cache=True)
def nll(params, t, T):
    mu, n, beta = params[0], params[1], params[2]
    A = 0.0
    ll = 0.0
    comp = 0.0
    prev = t[0]
    for i in range(t.shape[0]):
        if i > 0:
            A = np.exp(-beta * (t[i] - prev)) * (1.0 + A)
        lam = mu + n * beta * A
        ll += np.log(lam)
        comp += 1.0 - np.exp(-beta * (T - t[i]))
        prev = t[i]
    return -(ll - mu * T - n * comp)

def fit(t, T, starts=((0.5, 0.3, 1.0), (0.3, 0.6, 3.0))):
    """MLE in (log mu, logit n, log beta). Returns (mu, n, beta, nll, hit_bound)."""
    BMAX = 1e4
    def f(z):
        mu = np.exp(z[0]); n = 1/(1+np.exp(-z[1])); b = np.exp(z[2])
        n = min(n, 0.9999)
        return nll(np.array([mu, n, b]), t, T)
    rate = len(t) / T
    best = None
    for (m0, n0, b0) in starts:
        z0 = np.array([np.log(max(rate*(1-n0), 1e-6)), np.log(n0/(1-n0)), np.log(b0)])
        r = minimize(f, z0, method="L-BFGS-B",
                     bounds=[(-12, 8), (-8, 9), (np.log(1e-3), np.log(BMAX))])
        if best is None or r.fun < best.fun:
            best = r
    mu = float(np.exp(best.x[0])); n = float(1/(1+np.exp(-best.x[1]))); b = float(np.exp(best.x[2]))
    return mu, n, b, float(best.fun), bool(b > 0.99 * BMAX)

# ---- timestamp corruptions -------------------------------------------------
def round_down(t, delta):
    return np.floor(t / delta) * delta

def dither(t_rounded, delta, rng):
    return np.sort(t_rounded + rng.random(len(t_rounded)) * delta)

def spread(t_rounded, delta):
    """Evenly spread k tied events across their bin: positions (j+0.5)/k * delta."""
    out = t_rounded.copy()
    u, idx, cnt = np.unique(t_rounded, return_index=True, return_counts=True)
    for s, c in zip(idx, cnt):
        if c > 1:
            out[s:s+c] = t_rounded[s] + (np.arange(c) + 0.5) / c * delta
    return out

def tie_fraction(t):
    return float(np.mean(np.diff(t) == 0)) if len(t) > 1 else 0.0

def add_noise(t, sigma, rng):
    return np.sort(t + rng.normal(0, sigma, len(t)))
