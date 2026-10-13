import numpy as np, json, time
from multiprocessing import Pool
from hawkes import *
MU, BETA, T = 1.0, 1.0, 3000.0
NS = [0.3, 0.5, 0.8]
DELTAS = [0.15, 0.2, 0.25, 0.3, 0.4, 0.6]
REPS = 80
def one(args):
    n, rep = args
    seed = 5_000_000 + int(1000*n) * 1000 + rep
    ts = simulate(MU, n, BETA, T, seed)
    rng = np.random.default_rng(seed + 7)
    rows = []
    for d in DELTAS:
        tr = round_down(ts, d); tie = tie_fraction(tr)
        for name, x in (("raw", tr), ("dither", dither(tr, d, rng))):
            r = fit(x, T)
            rows.append(dict(n=n, rep=rep, level=d, method=name, tie=tie, nh=r[1], b=r[2], mu=r[0], hit=r[4]))
    return rows
if __name__ == "__main__":
    jobs = [(n, r) for n in NS for r in range(REPS)]
    t0 = time.time()
    with Pool(2) as p:
        out = p.map(one, jobs, chunksize=2)
    json.dump([r for rr in out for r in rr], open("../results/synthetic_fine.json", "w"))
    print("done", time.time()-t0)
