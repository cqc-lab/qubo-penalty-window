"""Analyse qpu_interleave.py: paired effect of chain length on success and on the direct-fit beta_eff.

Uses the beta_fine.py emulation grid (sigma = 0, beta step 0.1), which covers the main-run instances on the main
lambda grid. Instances are resampled jointly for both embeddings, so the interval on the beta difference is paired.

    python interleave_analyze.py qpu_interleave.csv --grid beta_fine_grid.csv
"""
import argparse

import numpy as np
import pandas as pd


def fit(S, C):
    """beta with the smallest pooled RMSE, given per-instance sums S and counts C (instances x beta)."""
    return S.columns.to_numpy()[(S.sum() / C.sum()).to_numpy().argmin()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="qpu_interleave.csv")
    ap.add_argument("--grid", default="beta_fine_grid.csv")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="interleave_beta.csv")
    args = ap.parse_args()

    q = pd.read_csv(args.csv)
    q["success_qpu"] = (q.succ_a + q.succ_b) / (q.reads_a + q.reads_b)
    q["loglam"] = q.loglam.round(3)
    g = pd.read_csv(args.grid)[["inst_id", "beta", "loglam", "success"]]
    g["loglam"] = g.loglam.round(3)
    d = q.merge(g, on=["inst_id", "loglam"])
    d["se"] = (d.success - d.success_qpu) ** 2
    rng = np.random.default_rng(0)

    rows = []
    for n, h in d.groupby("n"):
        per = {e: h[h.emb == e].groupby(["inst_id", "beta"]).se.agg(["sum", "count"]) for e in ("short", "long")}
        ids = np.intersect1d(per["short"].index.get_level_values(0).unique(),
                             per["long"].index.get_level_values(0).unique())
        S = {e: per[e]["sum"].unstack("beta").loc[ids] for e in per}
        C = {e: per[e]["count"].unstack("beta").loc[ids] for e in per}
        best = {e: fit(S[e], C[e]) for e in per}
        succ = q[q.n == n].pivot_table(index="inst_id", columns="emb", values="success_qpu").loc[ids]
        dsucc = (succ["short"] - succ["long"]).to_numpy()
        bs_b, bs_d, bs_s = {e: [] for e in per}, [], []
        for _ in range(args.boot):
            pick = rng.integers(len(ids), size=len(ids))
            b = {e: fit(S[e].iloc[pick], C[e].iloc[pick]) for e in per}
            for e in per:
                bs_b[e].append(b[e])
            bs_d.append(b["short"] - b["long"])
            bs_s.append(dsucc[pick].mean())
        chain = q[q.n == n].groupby("emb").chain_len.first()
        cb = q[q.n == n].groupby("emb").chain_break.mean()
        for e in ("short", "long"):
            rows.append(dict(n=n, emb=e, chain_len=chain[e], beta=best[e], lo=np.quantile(bs_b[e], 0.025),
                             hi=np.quantile(bs_b[e], 0.975), success=succ[e].mean(), chain_break=cb[e],
                             instances=len(ids)))
        rows.append(dict(n=n, emb="short-long", beta=best["short"] - best["long"],
                         lo=np.quantile(bs_d, 0.025), hi=np.quantile(bs_d, 0.975), success=dsucc.mean(),
                         success_lo=np.quantile(bs_s, 0.025), success_hi=np.quantile(bs_s, 0.975),
                         short_better=int((dsucc > 0).sum()), instances=len(ids)))
    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False)
    print("direct fit (sigma = 0, beta step 0.1), 95% bootstrap intervals over instances (paired for short-long);")
    print("success = mean over the lambda grid, short-long paired per instance:")
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
