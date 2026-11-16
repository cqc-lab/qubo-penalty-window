"""Paired test of the hardware scale: the two programmings of a QPU point differ only in their gauge.

With the extended coupling range the gauge changes the auto-scale factor, so some pairs of programmings of the
same (instance, lambda) ran at different hardware scales. If the QPU samples at a fixed inverse temperature in
programmed units, the programming with the smaller factor (stronger problem couplers in hardware units) should
succeed more often, by the amount the SA master curve predicts for beta_eff g / s_auto.

Reads embedded_gauge.csv (embedded_gauge.py); uses points with lambda >= 2 lambda_exact, default chain strength,
and pairs whose factors differ by more than 10 percent.

    python gauge_pairs.py --beta 2.9
"""
import argparse

import numpy as np
import pandas as pd

from embedded_scale import master_curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="embedded_gauge.csv")
    ap.add_argument("--beta", type=float, default=2.9)
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--out", default="gauge_pairs.csv")
    args = ap.parse_args()
    pred = master_curve("report/scaling_points.csv")
    f = pd.read_csv(args.csv)
    rel = f.loglam - np.log10(f.lam_exact.where(f.lam_exact > 0))
    f = f[(rel >= np.log10(2)) & (f.setting == "default") & f.gap.notna()]
    rows, n_points = [], 0
    for (run, row), g in f.groupby(["run", "row"]):
        if len(g) != 2:
            continue
        n_points += 1
        lo, hi = sorted([g.iloc[0], g.iloc[1]], key=lambda r: r.s_auto)
        if lo.s_auto / hi.s_auto < 0.9:
            rows.append(dict(run=run, inst_id=lo.inst_id, loglam=lo.loglam, ratio=lo.s_auto / hi.s_auto,
                             success_small=lo.success, success_large=hi.success, obs=lo.success - hi.success,
                             pred=pred(np.log10(args.beta * lo.gap / lo.s_auto))
                             - pred(np.log10(args.beta * hi.gap / hi.s_auto))))
    o = pd.DataFrame(rows)
    o.to_csv(args.out, index=False)
    rng = np.random.default_rng(0)
    ids = o.inst_id.unique()
    diff, slope = [], []
    for _ in range(args.boot):
        oo = pd.concat([o[o.inst_id == i] for i in rng.choice(ids, len(ids))])
        diff.append(oo.obs.mean())
        slope.append((oo.obs * oo.pred).sum() / (oo.pred ** 2).sum())
    print(f"{len(o)} of {n_points} points have factors differing by >10% (median ratio {o.ratio.median():.2f}), "
          f"{len(ids)} instances")
    print(f"success, smaller-factor minus larger-factor programming: {o.obs.mean():+.4f} "
          f"(95% CI {np.quantile(diff, 0.025):+.4f} to {np.quantile(diff, 0.975):+.4f}); "
          f"relative {o.obs.sum() / o.success_large.sum():+.3f}")
    print(f"predicted for beta fixed in programmed units: {o.pred.mean():+.4f}; in logical units: 0")
    print(f"slope observed ~ predicted: {(o.obs * o.pred).sum() / (o.pred ** 2).sum():.2f} "
          f"(95% CI {np.quantile(slope, 0.025):.2f} to {np.quantile(slope, 0.975):.2f})")


if __name__ == "__main__":
    main()
