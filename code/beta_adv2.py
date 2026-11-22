"""Direct fit of beta_eff for the Advantage2 run (qpu_adv2_main.csv), as beta_fine.py does for Advantage.

Advantage2_system1 programs h in [-6, 6] instead of [-4, 4], so the emulation rescales with h_range 6 (J range 1,
as before). The main instances are emulated at sigma = 0 on a beta grid of step 0.1 from 1 to 10 (beta_fine.py stops at 6), and beta_eff per size
is the beta minimising the RMSE of the success curves, with 95% bootstrap intervals over instances.

    python beta_adv2.py --workers 30
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

import gonogo
import qpu_beta

gonogo.H_RANGE = 6.0  # Advantage2 h range; read at call time by gonogo.sample_condition
qpu_beta.BETAS = list(np.round(np.arange(1.0, 10.0001, 0.1), 2))  # up to 10: Advantage2 is colder
qpu_beta.SIGMAS = [0.0]
for b in qpu_beta.BETAS:
    gonogo.CONDITIONS[f"b{b}_s0.0"] = (True, (0.1, b), 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--qpu", default="qpu_adv2_main.csv")
    ap.add_argument("--grid", default="beta_adv2_grid.csv")
    ap.add_argument("--out", default="beta_adv2.csv")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    have = set(pd.read_csv(args.grid).beta.round(2)) if os.path.exists(args.grid) else set()
    missing = [b for b in qpu_beta.BETAS if round(b, 2) not in have]
    if missing:  # emulate only the betas not yet in the grid file, and append them
        qpu_beta.BETAS = missing
        loglams = sorted(np.round(np.linspace(0.5, 2.5, 15), 3))
        tasks = [(n, max(1, round(rho * n)), idx, loglams) for n in [10, 14, 18] for rho in [0.5, 1.0]
                 for idx in range(8)]
        with Pool(args.workers) as pool:
            new = pd.DataFrame([r for rs in pool.map(qpu_beta.emulate, tasks) for r in rs])
        new.to_csv(args.grid, mode="a", index=False, header=not os.path.exists(args.grid))
    if not os.path.exists(args.qpu):
        print("grid written; QPU file not present yet")
        return
    q = pd.read_csv(args.qpu)
    q["success_qpu"] = qpu_beta.rate(q)
    q["loglam"] = q.loglam.round(3)
    g = pd.read_csv(args.grid)[["inst_id", "beta", "loglam", "success"]]
    g["loglam"] = g.loglam.round(3)
    d = q[["n", "inst_id", "loglam", "success_qpu"]].merge(g, on=["inst_id", "loglam"])
    d["se"] = (d.success - d.success_qpu) ** 2
    rng = np.random.default_rng(0)
    rows = []
    for n, h in d.groupby("n"):
        per = h.groupby(["inst_id", "beta"]).se.agg(["sum", "count"])
        ids = per.index.get_level_values(0).unique().to_numpy()
        S = per["sum"].unstack("beta").loc[ids]
        C = per["count"].unstack("beta").loc[ids]
        betas = S.columns.to_numpy()
        best = betas[(S.sum() / C.sum()).to_numpy().argmin()]
        bs = [betas[(S.iloc[p].sum() / C.iloc[p].sum()).to_numpy().argmin()]
              for p in (rng.integers(len(ids), size=len(ids)) for _ in range(args.boot))]
        rows.append(dict(run="adv2", n=n, beta=best, lo=np.quantile(bs, 0.025), hi=np.quantile(bs, 0.975),
                         rmse=float(np.sqrt(S[best].sum() / C[best].sum())), instances=len(ids)))
    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False)
    print("Advantage2 direct fit (sigma = 0, beta step 0.1, h range 6):")
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
