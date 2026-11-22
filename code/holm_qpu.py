"""Multiple-comparison check of the paired hardware comparisons reported in the paper.

Every comparison is a paired difference per instance. Two-sided p-values come from a sign-flip randomisation
test on the per-instance differences; Holm's step-down procedure is applied across all of them together.

  selection   qpu_estimator.csv   est vs every other method                  (Table 6)
  published   qpu_garcia.csv      est (re-scored) vs the methods of Garcia    (Table C2)
  one-hot     qpu_c2.csv          est vs every other method                   (Table C1)
  large       qpu_large.csv       est vs every other method, near5            (Table 8)
  chain       qpu_interleave.csv  short - long chains, per n
  strength    qpu_c1.csv          scaled - default chain strength, per n

    python holm_qpu.py --dir . --out holm_qpu.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

FLIPS = 100_000


def p_signflip(d, rng):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(FLIPS, len(d)))
    return (np.sum(np.abs((signs * d).mean(1)) >= obs - 1e-12) + 1) / (FLIPS + 1)


def paired(df, base, value, idx="inst_id"):
    piv = df.pivot_table(index=idx, columns="method", values=value)
    return {m: piv[base] - piv[m] for m in piv.columns if m != base}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--out", default="holm_qpu.csv")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    path = lambda f: os.path.join(args.dir, f)
    tests = []

    for fam, f, base in [("selection", "qpu_estimator.csv", "est"), ("published", "qpu_garcia.csv", "est_re"),
                         ("one-hot", "qpu_c2.csv", "est")]:
        for m, d in paired(pd.read_csv(path(f)), base, "success").items():
            if m != "oracle":
                tests.append((fam, f"{base} - {m}", d))
    lg = pd.read_csv(path("qpu_large.csv"))
    lg = lg[(lg.family == "sparse") & (lg.kind == "method")].assign(inst_id=lambda x: x.n.astype(str) + "_" +
                                                                    x.M.astype(str) + "_" + x.idx.astype(str))
    for m, d in paired(lg, "est", "near5").items():
        tests.append(("large", f"est - {m}", d))
    il = pd.read_csv(path("qpu_interleave.csv"))
    il["s"] = (il.succ_a + il.succ_b) / (il.reads_a + il.reads_b)
    for n, g in il.groupby("n"):
        p = g.pivot_table(index="inst_id", columns="emb", values="s")
        tests.append(("chain", f"short - long, n={n}", p["short"] - p["long"]))
    c1 = pd.read_csv(path("qpu_c1.csv"))
    c1["s"] = (c1.succ_a + c1.succ_b) / (c1.reads_a + c1.reads_b)
    for n, g in c1.groupby("n"):
        p = g.pivot_table(index="inst_id", columns="setting", values="s")
        tests.append(("strength", f"scaled - default, n={n}", p["scaled"] - p["default"]))

    rows = [dict(family=fam, comparison=name, n_inst=int(d.notna().sum()), mean_diff=float(d.mean()),
                 p=p_signflip(d, rng)) for fam, name, d in tests]
    res = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    k = len(res)
    adj = np.maximum.accumulate((k - np.arange(k)) * res.p.to_numpy())
    res["p_holm"] = np.minimum(adj, 1.0)
    res["sig_raw"], res["sig_holm"] = res.p < 0.05, res.p_holm < 0.05
    res.to_csv(args.out, index=False)
    pd.set_option("display.width", 160)
    print(res.round(4).to_string(index=False))
    print(f"\n{k} tests; significant at 0.05: {res.sig_raw.sum()} raw, {res.sig_holm.sum()} after Holm")
    lost = res[res.sig_raw & ~res.sig_holm]
    print("lost under Holm:", lost[["family", "comparison", "mean_diff", "p", "p_holm"]].to_dict("records"))


if __name__ == "__main__":
    main()
