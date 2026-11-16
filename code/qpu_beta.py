"""Effective temperature of the QPU, and whether the scaling law carries over to it.

1. Direct fit: emulate the qpu_main instances on a (beta, sigma) grid and pick the pair whose success curves
   are closest to the QPU's (RMSE; fit on even-indexed instances, report held-out RMSE on odd ones).
2. Scaling-law fit: take the SA master curve f(log10(beta g / s)) from results.csv (sigma = 0.02, beta in
   {2, 5, 10}) and find the single beta_eff for which f also predicts the QPU points. If the law holds,
   this beta_eff agrees with the direct fit and the collapse R^2 on QPU points is close to SA's.

    python qpu_beta.py --qpu qpu_main.csv --sa results.csv --workers 30
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

import gonogo
import scaling

BETAS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
SIGMAS = [0.0, 0.02, 0.05]
SEED = 2026
for b in BETAS:
    for s in SIGMAS:
        gonogo.CONDITIONS[f"b{b}_s{s}"] = (True, (0.1, b), s)


def emulate(task):
    from dwave.samplers import SimulatedAnnealingSampler
    sampler = SimulatedAnnealingSampler()
    n, M, idx, loglams = task
    seed = int(np.random.SeedSequence([SEED, n, M, idx]).generate_state(1)[0])
    inst = gonogo.make_instance(n, M, seed)
    rng = np.random.default_rng(seed + 17)
    w = 1 << np.arange(n)
    rows = []
    for b in BETAS:
        for s in SIGMAS:
            for ll in loglams:
                h, J = gonogo.qubo_to_ising(inst["U"] + 10.0 ** ll * inst["P"])
                S = np.concatenate(gonogo.sample_condition(h, J, f"b{b}_s{s}",
                                                           dict(n_prog=2, reads=250, sweeps=1000), rng, sampler))
                x = ((S + 1) // 2) @ w
                f = inst["feas"][x]
                rows.append(dict(inst_id=f"n{n}_M{M}_{idx}", n=n, idx=idx, beta=b, sigma=s, loglam=ll,
                                 success=(f & (inst["E"][x] == inst["E_star"])).mean(), feas=f.mean()))
    return rows


def rate(df):
    return (df.succ_a + df.succ_b) / (df.reads_a + df.reads_b)


def features(df):
    feats = []
    for inst_id in df.inst_id.unique():
        n, M, idx = (int(t[1:]) if i < 2 else int(t) for i, t in enumerate(inst_id.split("_")))
        feats += scaling.instance_features(n, M, idx, SEED, np.sort(df.loglam.unique()))
    return df.merge(pd.DataFrame(feats), on=["inst_id", "loglam"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qpu", default="qpu_main.csv")
    ap.add_argument("--sa", default="results.csv")
    ap.add_argument("--out", default="qpu_beta_grid.csv")
    ap.add_argument("--workers", type=int, default=30)
    args = ap.parse_args()

    q = pd.read_csv(args.qpu)
    q["success"], q["feas"] = rate(q), (q.feas_a + q.feas_b) / (q.reads_a + q.reads_b)
    q["idx"] = q.inst_id.str.split("_").str[-1].astype(int)

    # ---- 1. direct fit on a (beta, sigma) grid
    if not os.path.exists(args.out):
        tasks = [(n, M, int(iid.split("_")[-1]), sorted(g.loglam.unique()))
                 for (n, M, iid), g in q.groupby(["n", "M", "inst_id"])]
        with Pool(args.workers) as pool:
            pd.DataFrame([r for rs in pool.map(emulate, tasks) for r in rs]).to_csv(args.out, index=False)
    g = pd.read_csv(args.out).merge(q[["inst_id", "loglam", "success", "feas"]], on=["inst_id", "loglam"],
                                    suffixes=("", "_qpu"))
    g["se"] = (g.success - g.success_qpu) ** 2
    g["fe"] = (g.feas - g.feas_qpu) ** 2
    g["half"] = np.where(g.idx % 2 == 0, "fit", "holdout")
    tab = g.groupby(["half", "beta", "sigma"])[["se", "fe"]].mean().pow(0.5).unstack("half")
    tab.columns = [f"{a}_{b}" for a, b in tab.columns]
    tab = tab.rename(columns=lambda c: c.replace("se_", "rmse_succ_").replace("fe_", "rmse_feas_"))
    print("1. direct fit: RMSE vs QPU (success and feasible fraction), fit = even instances, holdout = odd")
    print(tab.round(3).to_string())
    best = tab.rmse_succ_fit.idxmin()
    print(f"best (beta, sigma) on fit half: {best}, holdout RMSE {tab.loc[best, 'rmse_succ_holdout']:.3f}")
    per_n = g.groupby(["n", "beta", "sigma"]).se.mean().pow(0.5)
    print("best (beta, sigma) per n (all instances):",
          {n: per_n.loc[n].idxmin() for n in sorted(g.n.unique())})

    # ---- 2. scaling-law fit: SA master curve, then one beta_eff for the QPU
    sa = pd.read_csv(args.sa)
    sa = sa[sa.cond.isin(["S1_beta2", "C3_ice02", "S2_beta10"])].copy()
    sa["success"] = rate(sa)
    sa = features(sa)
    sa["beta"] = sa.cond.map({c: gonogo.CONDITIONS[c][1][1] for c in sa.cond.unique()})
    qf = features(q)
    for d in (sa, qf):
        d["x_rel"] = d.loglam - np.log10(d.lam_exact.where(d.lam_exact > 0))
    sa, qf = (d[(d.x_rel >= np.log10(2)) & d.gap.notna()].copy() for d in (sa, qf))
    sa["x"] = np.log10(sa.beta * sa.gap / sa.scale)
    width = 0.1
    master = sa.groupby(np.floor(sa.x / width)).success.mean()

    def predict(x):
        b = np.floor(x / width)
        return np.interp(b, master.index.to_numpy(), master.to_numpy())

    y = qf.success.to_numpy()
    sst = np.sum((y - y.mean()) ** 2)
    print(f"\n2. scaling law: SA master curve from {len(sa)} SA points; {len(qf)} QPU points at lambda >= 2 lambda_exact")
    fits = []
    for b in np.round(np.arange(0.5, 6.01, 0.1), 2):
        x = np.log10(b * qf.gap / qf.scale)
        fits.append((b, 1 - np.sum((y - predict(x)) ** 2) / sst))
    fits = pd.DataFrame(fits, columns=["beta", "r2"]).set_index("beta").r2
    beta_eff = fits.idxmax()
    print(f"beta_eff (max predictive R^2 of the SA master curve on QPU points): {beta_eff}, R^2 = {fits.max():.3f}")
    print("predictive R^2 at beta =", {b: round(fits.loc[b], 3) for b in [1.0, 1.5, 2.0, 2.5, 3.0, 5.0]})
    for n in sorted(qf.n.unique()):
        d = qf[qf.n == n]
        yy = d.success.to_numpy()
        r = {b: 1 - np.sum((yy - predict(np.log10(b * d.gap / d.scale))) ** 2) / np.sum((yy - yy.mean()) ** 2)
             for b in fits.index}
        print(f"  n={n}: beta_eff {max(r, key=r.get)}, R^2 {max(r.values()):.3f}")

    qf["x_lam"], qf["x_therm"] = qf.loglam, np.log10(beta_eff * qf.gap / qf.scale)
    print("\nQPU points alone, binned-mean R^2 (as in scaling.py):")
    for col in ["x_lam", "x_rel", "x_therm"]:
        print(f"  {col:8s} {scaling.binned_r2(qf[col].to_numpy(), y, 0.1):.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    ax.plot(fits.index, fits.values, color="#2a78d6", lw=2)
    ax.axvline(beta_eff, color="#898781", lw=1, ls="--")
    ax.set_xlabel(r"$\beta_{\mathrm{eff}}$")
    ax.set_ylabel(r"predictive $R^2$ of SA master curve on QPU")
    ax = axes[1]
    xs = master.index.to_numpy() * width + width / 2
    ax.plot(xs, master.to_numpy(), color="#898781", lw=2, label=r"SA master curve ($\beta$=2,5,10)")
    for n, c in zip(sorted(qf.n.unique()), ["#2a78d6", "#eb6834", "#1baf7a"]):
        d = qf[qf.n == n]
        m = d.groupby(np.round(d.x_therm * 5) / 5).success.mean()
        ax.plot(m.index, m.values, "o", color=c, ms=6, mec="white", mew=1.5, label=f"QPU, n={n}")
    ax.set_xlabel(r"$\log_{10}(\beta_{\mathrm{eff}}\, g / s(\lambda))$")
    ax.set_ylabel(r"success rate ($\lambda \geq 2\lambda_{\mathrm{exact}}$)")
    ax.legend(fontsize=8, frameon=False)
    for a in axes:
        a.grid(color="#e4e3dd", lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig("qpu_scaling_collapse.png", dpi=150)


if __name__ == "__main__":
    main()
