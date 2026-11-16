"""Direct fit of beta_eff on a fine grid, with bootstrap intervals over instances.

Emulates the QPU instances (main and bign; the chain run reuses the main instances) at sigma = 0 on a fine
beta grid, then for every QPU run and size picks the beta minimising the RMSE of the success curves, and
resamples instances to get a 95% interval. Coarse-grid version: qpu_beta.py.

    python beta_fine.py --workers 30
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

import gonogo
import qpu_beta

qpu_beta.BETAS = list(np.round(np.arange(1.0, 6.0001, 0.1), 2))
qpu_beta.SIGMAS = [0.0]
for b in qpu_beta.BETAS:
    gonogo.CONDITIONS[f"b{b}_s0.0"] = (True, (0.1, b), 0.0)
RUNS = {"main": "qpu_main.csv", "chain": "qpu_chain.csv", "bign": "qpu_bign.csv"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--out", default="beta_fine_grid.csv")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    q = pd.concat([pd.read_csv(p).assign(run=r) for r, p in RUNS.items()])
    q["success_qpu"] = qpu_beta.rate(q)
    if not os.path.exists(args.out):
        inst = q[q.run != "chain"].groupby(["n", "M", "inst_id"]).loglam.unique()
        tasks = [(n, M, int(iid.split("_")[-1]), sorted(ll)) for (n, M, iid), ll in inst.items()]
        with Pool(args.workers) as pool:
            pd.DataFrame([r for rs in pool.map(qpu_beta.emulate, tasks) for r in rs]).to_csv(args.out, index=False)
    g = pd.read_csv(args.out)[["inst_id", "beta", "loglam", "success"]]
    d = q[["run", "n", "inst_id", "loglam", "success_qpu"]].merge(g, on=["inst_id", "loglam"])
    d["se"] = (d.success - d.success_qpu) ** 2
    rng = np.random.default_rng(0)
    rows = []
    for (run, n), h in d.groupby(["run", "n"]):
        per = h.groupby(["inst_id", "beta"]).se.agg(["sum", "count"])  # per-instance sums, resampled below
        ids = per.index.get_level_values(0).unique().to_numpy()
        S = per["sum"].unstack("beta").loc[ids]
        C = per["count"].unstack("beta").loc[ids]
        betas = S.columns.to_numpy()
        best = betas[(S.sum() / C.sum()).to_numpy().argmin()]
        bs = []
        for _ in range(args.boot):
            pick = rng.integers(len(ids), size=len(ids))
            bs.append(betas[(S.iloc[pick].sum() / C.iloc[pick].sum()).to_numpy().argmin()])
        rows.append(dict(run=run, n=n, beta=best, lo=np.quantile(bs, 0.025), hi=np.quantile(bs, 0.975),
                         rmse=float(np.sqrt(S[best].sum() / C[best].sum())), instances=len(ids)))
    res = pd.DataFrame(rows)
    res.to_csv("beta_fine.csv", index=False)
    print("direct fit (sigma = 0, beta step 0.1) with 95% bootstrap intervals over instances:")
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
