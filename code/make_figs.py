import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "serif"})
a = pd.DataFrame(json.load(open("../results/synthetic.json")))
b = pd.DataFrame(json.load(open("../results/synthetic_fine.json")))
rd = a[a.kind == "round"][["n","rep","level","method","tie","nh","b","hit"]].copy()
rd["src"] = "coarse"; b["src"] = "fine"
al = pd.concat([rd, b[["n","rep","level","method","tie","nh","b","hit"]].assign(src="fine")], ignore_index=True)
al["err"] = al.nh - al.n
al["collapsed"] = al.b > 100
C = {"raw": "#c0392b", "dither": "#1f6f8b", "spread": "#7f8c8d"}
L = {"raw": "raw (ties kept)", "dither": "uniform dither", "spread": "even spread"}

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.35), sharey=True)
for ax, n in zip(axs, (0.5, 0.8)):
    for m in ("raw", "spread", "dither"):
        s = al[(al.n == n) & (al.method == m)].groupby("level").err
        x = np.array(sorted(s.groups.keys()))
        mean = s.mean().loc[x].values
        lo = s.quantile(0.1).loc[x].values; hi = s.quantile(0.9).loc[x].values
        ax.plot(x, mean, "-o", ms=3, color=C[m], label=L[m], lw=1.2)
        ax.fill_between(x, lo, hi, color=C[m], alpha=0.12, lw=0)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xscale("log"); ax.set_xlabel(r"timestamp resolution $\delta\beta$ (bin width / decay time)")
    ax.set_title(f"true branching ratio $n={n}$")
axs[0].set_ylabel(r"error in $\hat n$ ($\hat n - n$)"); axs[0].legend(loc="lower left", frameon=False)
axs[0].set_ylim(-0.32, 0.25)
plt.tight_layout(); plt.savefig("../fig_bias.pdf"); plt.close()

fig, ax = plt.subplots(figsize=(3.3, 2.3))
cols = {0.3: "#e69f00", 0.5: "#0072b2", 0.8: "#009e73"}
for n in (0.3, 0.5, 0.8):
    s = al[(al.n == n) & (al.method == "raw")].groupby("level").collapsed.mean()
    ax.plot(s.index, s.values, "-o", ms=3, lw=1.2, color=cols[n], label=f"$n={n}$")
ax.axvline(0.3, color="k", ls=":", lw=0.8)
ax.set_xscale("log"); ax.set_xlabel(r"$\delta\beta$"); ax.set_ylabel(r"fraction of raw fits with $\hat\beta>100\beta$")
ax.legend(frameon=False, loc="upper left"); plt.tight_layout(); plt.savefig("../fig_collapse.pdf"); plt.close()

# ---- table pieces ----
def row(n, d):
    r = al[(al.n == n) & (al.level == d)]
    rw, dt, sp = (r[r.method == m] for m in ("raw", "dither", "spread"))
    return dict(n=n, d=d, tie=rw.tie.mean(), raw_bias=rw.err.mean(), raw_col=rw.collapsed.mean(),
                dit_bias=dt.err.mean(), dit_rmse=np.sqrt((dt.err**2).mean()), dit_b=dt.b.median(),
                sp_bias=(sp.err.mean() if len(sp) else np.nan), reps=len(rw))
rows = [row(n, d) for n in (0.5, 0.8) for d in (0.03, 0.1, 0.3, 1.0)]
tb = pd.DataFrame(rows); print(tb.round(3).to_string())
tb.to_json("../results/table1.json", orient="records")
ora = a[a.method == "oracle"].groupby("n").agg(bias=("nh", lambda x: 0), reps=("rep", "count"))
o = a[a.method == "oracle"].copy(); o["err"] = o.nh - o.n
print("oracle", o.groupby("n").err.agg(["mean", "std", "count"]).round(4))
nz = a[a.kind == "noise"].copy(); nz["err"] = nz.nh - nz.n
t2 = nz.groupby(["n", "level"]).agg(bias=("err", "mean"), sd=("nh", "std"), hit=("hit", "mean"), reps=("rep", "count")).round(3)
print(t2); t2.reset_index().to_json("../results/table2.json", orient="records")
print("coarse reps", rd.groupby("n").rep.nunique().to_dict(), "fine reps", b.groupby("n").rep.nunique().to_dict())
print("collapse by n,level raw:\n", al[al.method=="raw"].groupby(["n","level"]).collapsed.mean().round(3).unstack(0))
