"""Scaling-law check: does success above the threshold collapse onto beta * g / s(lambda)?

g        = gap between the best and second-best feasible objective values (QUBO units = Ising units)
s(lambda)= QPU auto-scale factor of the penalized Ising problem
beta     = final inverse temperature of the SA schedule (hardware units)

Compares how well three candidate variables predict the success rate (binned-mean R^2):
log10(lambda), log10(lambda / lambda_exact), log10(beta * g / s).

    python scaling.py --csv results.csv --outdir report
"""
import argparse
import os

import numpy as np
import pandas as pd

import gonogo


def instance_features(n, M, idx, seed_base, loglams):
    seed = int(np.random.SeedSequence([seed_base, n, M, idx]).generate_state(1)[0])
    inst = gonogo.make_instance(n, M, seed)
    fe = np.unique(inst["E"][inst["feas"]])
    gap = float(fe[1] - fe[0]) if len(fe) > 1 else np.nan
    rows = []
    for ll in loglams:
        h, J = gonogo.qubo_to_ising(inst["U"] + 10.0 ** ll * inst["P"])
        s = max(np.abs(h).max() / gonogo.H_RANGE, np.abs(J).max() / gonogo.J_RANGE)
        rows.append(dict(inst_id=f"n{n}_M{M}_{idx}", loglam=ll, gap=gap, scale=s))
    return rows


def binned_r2(x, y, width):
    b = np.floor(x / width)
    pred = pd.Series(y).groupby(b).transform("mean").to_numpy()
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results.csv")
    ap.add_argument("--outdir", default="report")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = df[df.cond.isin(["C3_ice02", "S1_beta2", "S2_beta10", "C1_scaled"])].copy()
    df["r"] = (df.succ_a + df.succ_b) / (df.reads_a + df.reads_b)
    loglams = np.sort(df.loglam.unique())
    feats = []
    for inst_id in df.inst_id.unique():
        n, M, idx = (int(t[1:]) if i < 2 else int(t) for i, t in enumerate(inst_id.split("_")))
        feats += instance_features(n, M, idx, args.seed, loglams)
    df = df.merge(pd.DataFrame(feats), on=["inst_id", "loglam"])

    beta = df.cond.map({c: gonogo.CONDITIONS[c][1][1] for c in df.cond.unique()})
    df["x_lam"] = df.loglam
    df["x_rel"] = df.loglam - np.log10(df.lam_exact.where(df.lam_exact > 0))
    df["x_therm"] = np.log10(beta * df.gap / df.scale)
    df.to_csv(os.path.join(args.outdir, "scaling_points.csv"), index=False)

    # Above-threshold side only: lambda at least 2x lambda_exact, where constraint violation no longer limits success.
    hi = df[(df.x_rel >= np.log10(2)) & df.gap.notna()]
    print(f"{len(hi)} points above 2*lambda_exact, {hi.inst_id.nunique()} instances")
    for cond_set, label in [(["C3_ice02", "S1_beta2", "S2_beta10"], "sigma=0.02, beta in {2,5,10}"),
                            (["C1_scaled"], "sigma=0, beta=5")]:
        d = hi[hi.cond.isin(cond_set)]
        print(f"\n{label}: binned-mean R^2 of success rate")
        for col in ["x_lam", "x_rel", "x_therm"]:
            print(f"  {col:8s} {binned_r2(d[col].to_numpy(), d.r.to_numpy(), 0.1):.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"S1_beta2": "#2a78d6", "C3_ice02": "#eb6834", "S2_beta10": "#1baf7a"}
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6), sharey=True)
    for ax, col, xlabel in zip(axes, ["x_lam", "x_rel", "x_therm"],
                               [r"$\log_{10}\lambda$", r"$\log_{10}(\lambda/\lambda_{\mathrm{exact}})$",
                                r"$\log_{10}(\beta g / s(\lambda))$"]):
        for cond, c in colors.items():
            for n, ls in zip(sorted(hi.n.unique()), ["-", "--", ":"]):
                d = hi[(hi.cond == cond) & (hi.n == n)]
                m = d.groupby(np.round(d[col] * 5) / 5).r.mean()
                ax.plot(m.index, m.values, color=c, ls=ls, lw=1.6,
                        label=f"{cond}, n={n}" if col == "x_therm" else None)
        ax.set_xlabel(xlabel)
        ax.grid(color="#e4e3dd", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("success rate (λ ≥ 2 λ_exact)")
    axes[-1].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "scaling_collapse.png"), dpi=150)


if __name__ == "__main__":
    main()
