"""Analyse qpu_large.py against the emulation of the same instances (large.py, large_beta.py).

  - methods (sparse): mean near5 per cell, paired differences est - method with 95% bootstrap intervals, and the
    probability of a solution within 1% of E_ref within 5000 reads in total (est charged its 1000 selection reads);
  - curves: location of the near5 peak in log10(lambda / lam_ref), per cell of the mean curve;
  - beta_eff: the emulated beta (large_beta.py: 0.75-2.5, sigma = 0; large.py: 5, C3) whose near5 curves are closest
    to the QPU curves (RMSE over instances and lambda), with a bootstrap over instances.

    python qpu_large_analyze.py qpu_large.csv --large large.csv --beta large_beta.csv
"""
import argparse

import numpy as np
import pandas as pd

METHODS = ["ref", "seq15", "seq", "est_nomargin", "grid", "inst", "wc"]
TOTAL, SELECT = 5000, 1000


def boot_mean(x, rng, B=2000):
    x = np.asarray(x)
    bs = x[rng.integers(len(x), size=(B, len(x)))].mean(1)
    return x.mean(), np.quantile(bs, 0.025), np.quantile(bs, 0.975)


def p_hit(p, reads):
    return 1 - (1 - p) ** reads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="qpu_large.csv")
    ap.add_argument("--large", default="large.csv")
    ap.add_argument("--beta", default="large_beta.csv")
    ap.add_argument("--out", default="qpu_large_summary.csv")
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    rng = np.random.default_rng(0)
    q = pd.read_csv(args.csv)
    key = ["family", "n", "M", "idx"]
    rows = []

    # ------------------------------------------------------------ methods (sparse)
    m = q[q.kind == "method"]
    if len(m):
        piv = m.pivot_table(index=key, columns="method", values="near5")
        near = m.pivot_table(index=key, columns="method", values="near")
        print("mean near5 by cell (QPU)")
        print(piv.groupby(level=["n", "M"]).mean()[["ref", "est"] + METHODS[1:]].round(3).to_string())
        print("\npaired est - method, near5, all sparse instances:")
        for meth in METHODS:
            d, lo, hi = boot_mean(piv["est"] - piv[meth], rng)
            rows.append(dict(part="diff", family="sparse", method=meth, value=d, lo=lo, hi=hi))
            print(f"  {meth:13s} {d:+.3f} ({lo:+.3f}, {hi:+.3f})")
        pe = p_hit(near["est"], TOTAL - SELECT)
        print("\nP(within 1% of E_ref, 5000 reads in total), mean over instances:")
        for meth in ["est", "ref", "inst", "wc"]:
            p = pe if meth == "est" else p_hit(near[meth], TOTAL)
            rows.append(dict(part="p1pct", family="sparse", method=meth, value=p.mean()))
            print(f"  {meth:5s} {p.mean():.2f}")
        rel = m[m.method == "est"].rel
        print(f"\nest log10(lambda / lam_ref) quantiles 10/50/90: {rel.quantile([0.1, 0.5, 0.9]).round(2).tolist()}"
              f" (target 0.18)")
        rows.append(dict(part="est_rel_median", family="sparse", method="est", value=rel.median()))
        print(f"feasible fraction at est: {m[m.method == 'est'].feas.mean():.3f}; E_ref improved on "
              f"{q.drop_duplicates(key).E_ref_improved.sum()} instances")

    # ------------------------------------------------------------ curves and beta_eff
    c = q[q.kind == "curve"].copy()
    c["rel"] = c.rel.round(2)
    emu = []
    lb = pd.read_csv(args.beta)
    emu.append(lb[lb.kind == "curve"][key + ["beta", "rel", "near5"]])
    lg = pd.read_csv(args.large)
    emu.append(lg[lg.kind == "curve"][key + ["rel", "near5"]].assign(beta=5.0))
    emu = pd.concat(emu)
    emu["rel"] = emu.rel.round(2)
    for fam, cf in c.groupby("family"):
        mean = cf.groupby(["n", "M", "rel"]).near5.mean().reset_index()
        peak = mean.loc[mean.groupby(["n", "M"]).near5.idxmax()]
        print(f"\n{fam}: peak of the mean near5 curve (QPU)")
        print(peak.rename(columns={"rel": "peak_rel", "near5": "peak_near5"}).round(3).to_string(index=False))
        for _, r in peak.iterrows():
            rows.append(dict(part="peak", family=fam, n=r.n, M=r.M, value=r.rel, near5=r.near5))
        d = cf[key + ["rel", "near5", "chain_len"]].merge(emu, on=key + ["rel"], suffixes=("_qpu", "_emu"))
        d["se"] = (d.near5_qpu - d.near5_emu) ** 2
        for n, dn in d.groupby("n"):
            per = dn.groupby(["idx", "M", "beta"]).se.agg(["sum", "count"])
            S, C = per["sum"].unstack("beta"), per["count"].unstack("beta")
            betas = S.columns.to_numpy()
            best = betas[(S.sum() / C.sum()).to_numpy().argmin()]
            bs = [betas[(S.iloc[p].sum() / C.iloc[p].sum()).to_numpy().argmin()]
                  for p in rng.integers(len(S), size=(1000, len(S)))]
            rmse = np.sqrt((S.sum() / C.sum()).to_numpy())
            chain = dn.drop_duplicates(["idx", "M"]).chain_len.mean()
            print(f"{fam} n={n}: chain {chain:.2f}; best beta {best:g} (95% {np.quantile(bs, 0.025):g}-"
                  f"{np.quantile(bs, 0.975):g}); RMSE by beta {dict(zip(betas, rmse.round(3)))}")
            rows.append(dict(part="beta", family=fam, n=n, chain_len=chain, value=best,
                             lo=np.quantile(bs, 0.025), hi=np.quantile(bs, 0.975)))
        print(f"{fam}: max near5 of any instance on the curve {cf.near5.max():.3f}; mean feasible fraction "
              f"for rel >= 0.1: {cf[cf.rel >= 0.1].feas.mean():.3f}")
        rows.append(dict(part="max_near5", family=fam, value=cf.near5.max()))
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
