"""Analysis of qpu_strength.csv: does success follow the programmed energy scale?

For lambda >= 10^0.3 lambda_exact (about 2 lambda_exact; rel >= 0.3), each programming is a point with success, gap g, the QPU auto-scale
factor s_auto of its gauge and the strength k of the problem terms. Two models are compared with the SA master curve:
  programmed : success = curve(log10(beta k g / s_auto))   (temperature fixed in programmed units)
  logical    : success = curve(log10(beta g / s_auto))     (k has no effect)
each with its best beta (predictive R^2, as in the scaling fit). The paired differences k = 0.5 and 0.7 minus
k = 1 within a gauge are compared with the programmed-unit prediction, with bootstrap intervals over instances.

    python strength_analyze.py --csv qpu_strength.csv
"""
import argparse

import numpy as np
import pandas as pd

import gonogo
from embedded_scale import BETAS, master_curve


def gap_of(iid):
    n, M, idx = (int(t[1:]) if i < 2 else int(t) for i, t in enumerate(iid.split("_")))
    inst = gonogo.make_instance(n, M, int(np.random.SeedSequence([2026, n, M, idx]).generate_state(1)[0]))
    fe = np.unique(inst["E"][inst["feas"]])
    return float(fe[1] - fe[0]) if len(fe) > 1 else np.nan


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="qpu_strength.csv")
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--out", default="strength_summary.csv")
    args = ap.parse_args()
    pred = master_curve("report/scaling_points.csv")
    d = pd.read_csv(args.csv)
    d["success"] = d.succ / d.reads
    gaps = {i: gap_of(i) for i in d.inst_id.unique()}
    d["gap"] = d.inst_id.map(gaps)
    d = d[(d.rel >= 0.3 - 1e-9) & d.gap.notna()].copy()  # lambda >= 10^0.3 = 2.0 lambda_exact
    y = d.success.to_numpy()
    res = {}
    for name, kcol in [("programmed", d.k.to_numpy()), ("logical", np.ones(len(d)))]:
        sc = [r2(y, pred(np.log10(b * kcol * d.gap.to_numpy() / d.s_auto.to_numpy()))) for b in BETAS]
        i = int(np.argmax(sc))
        res[name] = (BETAS[i], sc[i])
        print(f"{name:10s} model: best beta {BETAS[i]:.2f}, predictive R^2 {sc[i]:.3f}")
    beta = res["programmed"][0]
    d["pred"] = pred(np.log10(beta * d.k * d.gap / d.s_auto))
    base = d[d.k == 1.0].set_index(["inst_id", "rel", "gauge"])
    rows = []
    rng = np.random.default_rng(0)
    for k in sorted(d.k.unique()):
        if k == 1.0:
            continue
        x = d[d.k == k].set_index(["inst_id", "rel", "gauge"])
        diff = (x.success - base.success).dropna()
        pdiff = (x.pred - base.pred).reindex(diff.index)
        ids = diff.index.get_level_values(0).unique()
        bs_d, bs_s = [], []
        for _ in range(args.boot):
            pick = rng.choice(ids, len(ids))
            dd = pd.concat([diff.loc[i] for i in pick])
            pp = pd.concat([pdiff.loc[i] for i in pick])
            bs_d.append(dd.mean())
            bs_s.append((dd * pp).sum() / (pp ** 2).sum())
        slope = (diff * pdiff).sum() / (pdiff ** 2).sum()
        rows.append(dict(k=k, pairs=len(diff), instances=len(ids), obs=diff.mean(), lo=np.quantile(bs_d, 0.025),
                         hi=np.quantile(bs_d, 0.975), pred=pdiff.mean(), slope=slope,
                         slope_lo=np.quantile(bs_s, 0.025), slope_hi=np.quantile(bs_s, 0.975),
                         rel_change=diff.sum() / base.success.reindex(diff.index).sum()))
    out = pd.DataFrame(rows)
    out = pd.concat([out, pd.DataFrame([dict(k="model", obs=np.nan, pred=np.nan,
                                             slope=res["programmed"][1], slope_lo=res["logical"][1],
                                             lo=res["programmed"][0], hi=res["logical"][0])])], ignore_index=True)
    out.to_csv(args.out, index=False)
    pd.set_option("display.width", 200)
    print("\nsuccess at k minus k = 1, same gauge (lambda >= 2 lambda_exact):")
    print(out[out.k != "model"].round(4).to_string(index=False))
    print("\nmean success by n, rel and k (all rel):")
    a = pd.read_csv(args.csv)
    a["success"] = a.succ / a.reads
    print(a.pivot_table(index=["n", "rel"], columns="k", values="success").round(3).to_string())
    print("\nchain-break fraction by k:", a.groupby("k").chain_break.mean().round(4).to_dict())
    print("term setting s_auto:", a.auto_term.value_counts(normalize=True).round(3).to_dict())


if __name__ == "__main__":
    main()
