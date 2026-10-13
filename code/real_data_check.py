"""Run on YOUR real trade timestamps (not run by the authors of this draft; see paper Limitations).
Usage: python3 real_data_check.py trades.csv [time_col] [unit]
  trades.csv : any CSV with a timestamp column (default 'time'); unit = ms | us | s  (default ms)
Prints, for coarsening levels 1x, 10x, 100x, 1000x of the native resolution:
  tie fraction, and Hawkes MLE (mu, n, beta, beta-at-bound) for raw vs dithered timestamps.
Rule from the paper: if raw n_hat ~= tie fraction and beta_hat sits at the bound, the raw fit is degenerate."""
import sys, numpy as np, pandas as pd
from hawkes import fit, round_down, dither, tie_fraction

path = sys.argv[1]; col = sys.argv[2] if len(sys.argv) > 2 else "time"
unit = sys.argv[3] if len(sys.argv) > 3 else "ms"
scale = {"ms": 1e-3, "us": 1e-6, "s": 1.0}[unit]
x = np.sort(pd.read_csv(path)[col].to_numpy(dtype=float)) * scale
x = x - x[0]; T = float(x[-1]) + 1e-9
native = {"ms": 1e-3, "us": 1e-6, "s": 1.0}[unit]
rng = np.random.default_rng(0)
print(f"events={len(x)}  span={T:.1f}s  native tie fraction={tie_fraction(x):.3f}")
for mult in (1, 10, 100, 1000):
    d = native * mult
    xr = round_down(x, d)
    for name, y in (("raw", xr), ("dither", dither(xr, d, rng) if mult > 1 else dither(xr, d, rng))):
        # rescale time so that fitted beta is O(1): use seconds -> divide by 1 s
        mu, n, b, f, hit = fit(y, T)
        print(f"res={d*1e3:9.3f} ms  {name:6s} tie={tie_fraction(xr):.3f}  n_hat={n:.3f}  beta_hat={b:.4g}/s  mu={mu:.4g}/s  at_bound={hit}")
