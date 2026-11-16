"""Independent check of the large-n reference threshold: the same bisection as large.reference, with tabu
search instead of heavy SA deciding whether the penalized ground state is feasible.

    python tabu_ref.py --workers 30
"""
import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd

import large


def tabu_reference(task):
    from dwave.samplers import TabuSampler
    family, n, M, idx = task
    inst, seed = large.make_large(family, n, M, idx)
    sampler = TabuSampler()
    rng = np.random.default_rng(seed + 5)

    def ground_feasible(lam):
        h, J = large.gonogo.qubo_to_ising(inst["U"] + lam * inst["P"])
        bqm = large.gonogo.dimod.BinaryQuadraticModel.from_ising(
            {i: h[i] for i in range(n)}, {(i, j): J[i, j] for i in range(n) for j in range(i + 1, n) if J[i, j]})
        ss = sampler.sample(bqm, num_reads=20, timeout=200, seed=int(rng.integers(2**31 - 1)))
        S = ss.record.sample[:, np.argsort(np.asarray(list(ss.variables)))]
        E, V = large.evaluate_states(inst, S)
        return V[np.argmin(E + lam * V)] == 0, E[V == 0].min() if (V == 0).any() else np.inf

    best = np.inf
    lo, hi = 1e-2, 2 * inst["lam_wc"]
    ok, e = ground_feasible(hi)
    best = min(best, e)
    while not ok:
        hi *= 4
        ok, e = ground_feasible(hi)
        best = min(best, e)
    for _ in range(12):
        mid = np.sqrt(lo * hi)
        ok, e = ground_feasible(mid)
        best = min(best, e)
        if ok:
            hi = mid
        else:
            lo = mid
    return dict(family=family, n=n, M=M, idx=idx, lam_ref_tabu=hi, E_best_tabu=best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--ref", default="large.csv")
    args = ap.parse_args()
    ref = pd.read_csv(args.ref).groupby(["family", "n", "M", "idx"])[["lam_ref", "E_ref"]].first().reset_index()
    tasks = [tuple(r) for r in ref[["family", "n", "M", "idx"]].itertuples(index=False)]
    with Pool(args.workers) as pool:
        out = pd.DataFrame(pool.map(tabu_reference, tasks))
    d = ref.merge(out, on=["family", "n", "M", "idx"])
    d["dlog"] = np.log10(d.lam_ref_tabu / d.lam_ref)
    d.to_csv("tabu_ref.csv", index=False)
    print("log10(lam_ref_tabu / lam_ref_SA) by cell: median, max |.|; and tabu best vs E_ref")
    print(d.groupby(["family", "n", "M"]).agg(med=("dlog", "median"), maxabs=("dlog", lambda x: x.abs().max()),
                                              tabu_better=("E_best_tabu", lambda x: 0)).round(3).to_string())
    print("overall: median", round(d.dlog.median(), 3), " |dlog| < 0.05:", round((d.dlog.abs() < 0.05).mean(), 3),
          " tabu found a better feasible value than E_ref:", int((d.E_best_tabu < d.E_ref - 1e-9).sum()))


if __name__ == "__main__":
    main()
