"""Analysis of qpu_c1.csv: default vs scaled chain strength, paired by instance and lambda.

  - paired success difference (default - scaled) with a bootstrap interval over instances
  - scaling-fit beta_eff per n and setting, with the logical scale and (default only) the embedded scale;
    under "scaled" the two scales coincide by construction
  - window shape on the lambda / lambda_exact grid: peak location, w50, success above 10 and 100 lambda_exact
  - binned R^2 of the three candidate variables for lambda >= 2 lambda_exact

    python c1_analyze.py qpu_c1.csv
"""
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

import embedded_scale
import scaling


def main():
    q = pd.read_csv(sys.argv[1])
    q["success"] = (q.succ_a + q.succ_b) / (q.reads_a + q.reads_b)
    rng = np.random.default_rng(0)

    w = q.pivot_table(index=["inst_id", "rel"], columns="setting", values="success").reset_index()
    w["d"] = w.default - w.scaled
    print("paired success difference default - scaled (mean over lambda, then bootstrap over instances):")
    for n, g in w.assign(n=w.inst_id.str.extract(r"n(\d+)_")[0].astype(int)).groupby("n"):
        per = g.groupby("inst_id").d.mean().to_numpy()
        bs = [rng.choice(per, len(per)).mean() for _ in range(4000)]
        print(f"  n={n}: {per.mean():+.4f} [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}], "
              f"{len(per)} instances")

    tasks = [("c1", iid, sorted(g.loglam.unique()), str(int(iid.split("_")[0][1:]))) for iid, g in q.groupby("inst_id")]
    with Pool(30) as pool:
        f = pd.DataFrame([r for rs in pool.map(embedded_scale.features, tasks) for r in rs])
    d = q.merge(f[["inst_id", "loglam", "gap", "s_log", "s_emb"]], on=["inst_id", "loglam"])
    hi = d[(d.rel >= np.log10(2)) & d.gap.notna()].copy()
    predict = embedded_scale.master_curve("report/scaling_points.csv")
    rows = []
    for (n, setting), g in hi.groupby(["n", "setting"]):
        ids = g.inst_id.unique()
        for col in (["s_log", "s_emb"] if setting == "default" else ["s_log"]):
            b, r2 = embedded_scale.best_beta(g, col, predict)
            bs = []
            for _ in range(300):
                pick = rng.choice(ids, len(ids))
                bs.append(embedded_scale.best_beta(pd.concat([g[g.inst_id == i] for i in pick]), col, predict)[0])
            rows.append(dict(n=n, setting=setting, scale=col, beta=b, r2=round(r2, 3), lo=np.quantile(bs, .025),
                             hi=np.quantile(bs, .975), points=len(g)))
    res = pd.DataFrame(rows)
    res.to_csv("c1_beta.csv", index=False)
    print("\nscaling-fit beta_eff (lambda >= 2 lambda_exact), 95% bootstrap over instances:")
    print(res.to_string(index=False))

    print("\nbinned R^2 (width 0.1) for lambda >= 2 lambda_exact, per setting:")
    for setting, g in hi.groupby("setting"):
        b = res[(res.setting == setting) & (res.scale == "s_log")].beta.mean()
        x = {"log10 lambda": g.loglam, "log10 lambda/lex": g.rel, "log10 beta g/s": np.log10(b * g.gap / g.s_log)}
        print(f"  {setting}: " + ", ".join(f"{k} {scaling.binned_r2(v.to_numpy(), g.success.to_numpy(), 0.1):.3f}"
                                         for k, v in x.items()))

    print("\nwindow shape per n and setting (medians over instances):")
    out = []
    for (n, setting, iid), g in q.groupby(["n", "setting", "inst_id"]):
        g = g.sort_values("rel")
        x, y = g.rel.to_numpy(), g.success.to_numpy()
        k = y.argmax()
        half = y >= 0.5 * y[k]
        lo = k
        while lo > 0 and half[lo - 1]:
            lo -= 1
        hi_ = k
        while hi_ < len(y) - 1 and half[hi_ + 1]:
            hi_ += 1
        out.append(dict(n=n, setting=setting, peak_rel=x[k], peak=y[k], w50_grid=x[hi_] - x[lo],
                        at10=np.interp(1.0, x, y), at100=np.interp(2.0, x, y), at1000=np.interp(3.0, x, y)))
    o = pd.DataFrame(out)
    print(o.groupby(["n", "setting"]).median().round(3).to_string())
    print("\nmeans:")
    print(o.groupby(["n", "setting"])[["peak", "at10", "at100", "at1000"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
