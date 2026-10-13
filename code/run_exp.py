import numpy as np, json, sys, time
from multiprocessing import Pool
from hawkes import *

MU, BETA, T = 1.0, 1.0, 3000.0
NS = [0.5, 0.8]
DELTAS = [0.01, 0.03, 0.1, 0.3, 1.0]
SIGMAS = [0.03, 0.1, 0.3]
REPS = 200

def one(args):
    n, rep = args
    seed = int(1000*n) * 100000 + rep
    ts = simulate(MU, n, BETA, T, seed)
    rng = np.random.default_rng(seed + 7)
    rows = []
    o = fit(ts, T)
    rows.append(dict(n=n, rep=rep, kind="oracle", level=0.0, method="oracle", N=len(ts), tie=0.0,
                     mu=o[0], nh=o[1], b=o[2], hit=o[4]))
    for d in DELTAS:
        tr = round_down(ts, d)
        tie = tie_fraction(tr)
        for name, x in (("raw", tr), ("dither", dither(tr, d, rng)), ("spread", spread(tr, d))):
            r = fit(x, T)
            rows.append(dict(n=n, rep=rep, kind="round", level=d, method=name, N=len(ts), tie=tie,
                             mu=r[0], nh=r[1], b=r[2], hit=r[4]))
    for s in SIGMAS:
        x = add_noise(ts, s, rng)
        r = fit(x, T)
        rows.append(dict(n=n, rep=rep, kind="noise", level=s, method="noisy", N=len(ts), tie=0.0,
                         mu=r[0], nh=r[1], b=r[2], hit=r[4]))
    return rows

if __name__ == "__main__":
    jobs = [(n, r) for n in NS for r in range(REPS)]
    t0 = time.time()
    with Pool(2) as p:
        out = p.map(one, jobs, chunksize=4)
    rows = [r for rr in out for r in rr]
    json.dump(rows, open("../results/synthetic.json", "w"))
    print("done", len(rows), "rows", time.time() - t0, "s")
