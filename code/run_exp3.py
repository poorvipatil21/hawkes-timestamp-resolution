"""Robustness of the collapse finding to optimizer bound, start points and sample length."""
import numpy as np, json, time
from multiprocessing import Pool
from scipy.optimize import minimize
from hawkes import simulate, nll, round_down, dither, tie_fraction
MU, BETA = 1.0, 1.0
REPS = 40
NS = [0.5, 0.8]
DELTAS = [0.1, 0.3, 0.6]
CONFIGS = {  # name: (T, bound, starts)
 "base":        (3000.0, 1e4, ((0.5,0.3,1.0),(0.3,0.6,3.0))),
 "bound1e3":    (3000.0, 1e3, ((0.5,0.3,1.0),(0.3,0.6,3.0))),
 "bound1e5":    (3000.0, 1e5, ((0.5,0.3,1.0),(0.3,0.6,3.0))),
 "T1000":       (1000.0, 1e4, ((0.5,0.3,1.0),(0.3,0.6,3.0))),
 "far_start":   (3000.0, 1e4, ((0.2,0.1,0.1),(0.2,0.9,30.0))),
}
def fitc(t, T, bmax, starts):
    def f(z):
        return nll(np.array([np.exp(z[0]), min(1/(1+np.exp(-z[1])),0.9999), np.exp(z[2])]), t, T)
    rate = len(t)/T; best=None
    for (m0,n0,b0) in starts:
        z0 = np.array([np.log(max(rate*(1-n0),1e-6)), np.log(n0/(1-n0)), np.log(b0)])
        r = minimize(f, z0, method="L-BFGS-B", bounds=[(-12,8),(-8,9),(np.log(1e-3),np.log(bmax))])
        if best is None or r.fun < best.fun: best = r
    return float(1/(1+np.exp(-best.x[1]))), float(np.exp(best.x[2]))
def one(args):
    cfg, n, rep = args
    T, bmax, starts = CONFIGS[cfg]
    seed = 9_000_000 + int(1000*n)*1000 + rep
    ts = simulate(MU, n, BETA, T, seed); rng = np.random.default_rng(seed+7); rows=[]
    for d in DELTAS:
        tr = round_down(ts, d)
        for name, x in (("raw", tr), ("dither", dither(tr, d, rng))):
            nh, b = fitc(x, T, bmax, starts)
            rows.append(dict(cfg=cfg, n=n, rep=rep, level=d, method=name, nh=nh, b=b, collapsed=bool(b>100*BETA)))
    return rows
if __name__ == "__main__":
    jobs=[(c,n,r) for c in CONFIGS for n in NS for r in range(REPS)]
    t0=time.time()
    with Pool(2) as p: out=p.map(one, jobs, chunksize=2)
    json.dump([r for rr in out for r in rr], open("../results/robustness.json","w"))
    print("done", time.time()-t0)
