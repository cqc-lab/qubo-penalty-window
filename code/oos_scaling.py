"""Out-of-sample check of the scaling variable on the QPU, against control variables.

The SA master curve of each variable is built from results.csv (sigma = 0.02, beta in {2, 5, 10}, lambda >= 2
lambda_exact), exactly as in qpu_beta.py. For the variables with an inverse temperature, beta_eff is fitted on one
half of the QPU instances (maximum predictive R^2) and the R^2 is reported on the other half:

  therm : log10(beta g / s(lambda))           the scaling variable
  nog   : log10(beta g_med / s(lambda))       control: the instance gap g replaced by the median gap of the SA set
  rel   : log10(lambda / lambda_exact)        control without a fitted parameter
  lam   : log10(lambda)                       control without a fitted parameter

Splits: even/odd instance index both ways, and random halves of the instances (stratified by n).

    python oos_scaling.py --qpu qpu_main.csv --points report/scaling_points.csv
"""
import argparse

import numpy as np
import pandas as pd

import qpu_beta

WIDTH = 0.1
BETAS = np.round(np.arange(0.5, 6.01, 0.1), 2)


def master(x, y):
    m = pd.Series(np.asarray(y)).groupby(np.floor(np.asarray(x) / WIDTH)).mean()
    return lambda z: np.interp(np.floor(z / WIDTH), m.index.to_numpy(), m.to_numpy())


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qpu", default="qpu_main.csv")
    ap.add_argument("--points", default="report/scaling_points.csv")
    ap.add_argument("--splits", type=int, default=500)
    ap.add_argument("--out", default="oos_scaling.csv")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    sa = pd.read_csv(args.points)
    sa = sa[sa.cond.isin(["S1_beta2", "C3_ice02", "S2_beta10"]) & (sa.x_rel >= np.log10(2)) & sa.gap.notna()].copy()
    sa["beta"] = sa.cond.map({"S1_beta2": 2.0, "C3_ice02": 5.0, "S2_beta10": 10.0})
    g_med = float(sa.drop_duplicates("inst_id").gap.median())
    f = {"therm": master(np.log10(sa.beta * sa.gap / sa.scale), sa.r.to_numpy()),
         "nog": master(np.log10(sa.beta * g_med / sa.scale), sa.r.to_numpy()),
         "rel": master(sa.x_rel.to_numpy(), sa.r.to_numpy()),
         "lam": master(sa.loglam.to_numpy(), sa.r.to_numpy())}

    q = pd.read_csv(args.qpu)
    q["success"] = qpu_beta.rate(q)
    q = qpu_beta.features(q)
    q["x_rel"] = q.loglam - np.log10(q.lam_exact.where(q.lam_exact > 0))
    q = q[(q.x_rel >= np.log10(2)) & q.gap.notna()].reset_index(drop=True)
    q["idx"] = q.inst_id.str.split("_").str[-1].astype(int)
    y = q.success.to_numpy()
    # predictions for every beta, computed once
    P = {"therm": np.array([f["therm"](np.log10(b * q.gap / q.scale)) for b in BETAS]),
         "nog": np.array([f["nog"](np.log10(b * g_med / q.scale)) for b in BETAS])}
    fixed = {"rel": f["rel"](q.x_rel.to_numpy()), "lam": f["lam"](q.loglam.to_numpy())}
    print(f"{len(sa)} SA points, {len(q)} QPU points, {q.inst_id.nunique()} QPU instances; median SA gap {g_med:g}")

    def evaluate(fit_mask):
        hold = ~fit_mask
        out = {}
        for v, Pv in P.items():
            k = int(np.argmax([r2(y[fit_mask], p[fit_mask]) for p in Pv]))
            out[v] = (BETAS[k], r2(y[hold], Pv[k][hold]))
        for v, p in fixed.items():
            out[v] = (np.nan, r2(y[hold], p[hold]))
        return out

    rows = []
    for name, mask in [("fit even, test odd", q.idx.to_numpy() % 2 == 0), ("fit odd, test even", q.idx.to_numpy() % 2 == 1)]:
        for v, (b, r) in evaluate(mask).items():
            rows.append(dict(split=name, variable=v, beta_eff=b, r2_holdout=r))
    insts = q.drop_duplicates("inst_id")[["inst_id", "n"]]
    rand = {v: [] for v in list(P) + list(fixed)}
    for _ in range(args.splits):
        pick = set(insts.groupby("n").inst_id.apply(lambda s: s.sample(len(s) // 2, random_state=rng.integers(2**31)))
                   .explode())
        res = evaluate(q.inst_id.isin(pick).to_numpy())
        for v, (b, r) in res.items():
            rand[v].append((b, r))
    for v, lst in rand.items():
        a = np.array(lst, dtype=float)
        rows.append(dict(split=f"{args.splits} random halves", variable=v, beta_eff=np.nanmedian(a[:, 0]),
                         r2_holdout=np.median(a[:, 1]), r2_lo=np.quantile(a[:, 1], 0.025),
                         r2_hi=np.quantile(a[:, 1], 0.975)))
    d = np.array([a[1] for a in rand["therm"]]) - np.array([a[1] for a in rand["nog"]])
    rows.append(dict(split=f"{args.splits} random halves", variable="therm - nog", r2_holdout=np.median(d),
                     r2_lo=np.quantile(d, 0.025), r2_hi=np.quantile(d, 0.975), frac_positive=float((d > 0).mean())))
    for v, Pv in P.items():  # in-sample, all points
        k = int(np.argmax([r2(y, p) for p in Pv]))
        rows.append(dict(split="in-sample, all points", variable=v, beta_eff=BETAS[k], r2_holdout=r2(y, Pv[k])))
    for v, p in fixed.items():
        rows.append(dict(split="in-sample, all points", variable=v, r2_holdout=r2(y, p)))
    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False)
    pd.set_option("display.width", 160)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
