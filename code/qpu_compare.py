"""Compare QPU rows with the SA emulation conditions on the same instances and lambdas.

    python qpu_compare.py qpu_pilot.csv --out qpu_compare.csv --workers 8
"""
import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd

import gonogo

CONDS = ["C1_scaled", "C2_ice01", "C3_ice02", "C4_ice05", "S1_beta2", "S2_beta10"]
SEED = 2026


def run_instance(task):
    from dwave.samplers import SimulatedAnnealingSampler
    sampler = SimulatedAnnealingSampler()
    n, M, idx, loglams = task
    seed = int(np.random.SeedSequence([SEED, n, M, idx]).generate_state(1)[0])
    inst = gonogo.make_instance(n, M, seed)
    rng = np.random.default_rng(seed + 5)
    weights = 1 << np.arange(n)
    rows = []
    for cond in CONDS:
        for ll in loglams:
            h, J = gonogo.qubo_to_ising(inst["U"] + 10.0 ** ll * inst["P"])
            S = np.concatenate(gonogo.sample_condition(h, J, cond, dict(n_prog=2, reads=250, sweeps=1000),
                                                       rng, sampler))
            x = ((S + 1) // 2) @ weights
            f = inst["feas"][x]
            rows.append(dict(inst_id=f"n{n}_M{M}_{idx}", n=n, M=M, cond=cond, loglam=ll,
                             success=(f & (inst["E"][x] == inst["E_star"])).mean(), feas=f.mean(),
                             rel=ll - np.log10(inst["lam_exact"])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qpu")
    ap.add_argument("--out", default="qpu_compare.csv")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    q = pd.read_csv(args.qpu)
    q["success"] = (q.succ_a + q.succ_b) / (q.reads_a + q.reads_b)
    q["feas"] = (q.feas_a + q.feas_b) / (q.reads_a + q.reads_b)
    q["rel"] = q.loglam - np.log10(q.lam_exact)
    tasks = []
    for (n, M, iid), g in q.groupby(["n", "M", "inst_id"]):
        tasks.append((n, M, int(iid.split("_")[-1]), sorted(g.loglam.round(3).unique())))
    with Pool(args.workers) as pool:
        sa = pd.DataFrame([r for rs in pool.map(run_instance, tasks) for r in rs])
    q["cond"] = "QPU"
    df = pd.concat([q[sa.columns], sa], ignore_index=True)
    df.to_csv(args.out, index=False)

    print("mean success by lambda/lambda_exact bin")
    df["relbin"] = (df.rel / 0.25).round() * 0.25
    print(df.pivot_table(index=["n", "relbin"], columns="cond", values="success", aggfunc="mean")
          [["QPU"] + CONDS].round(3).to_string())
    print("\nmean feasible fraction by bin")
    print(df.pivot_table(index=["n", "relbin"], columns="cond", values="feas", aggfunc="mean")
          [["QPU"] + CONDS].round(3).to_string())

    # Per instance: peak location, peak height, and how well each SA condition tracks the QPU curve.
    out = []
    for (iid, cond), g in df.groupby(["inst_id", "cond"]):
        g = g.sort_values("loglam")
        sm = np.convolve(g.success, np.ones(3) / 3, mode="same")
        out.append(dict(inst_id=iid, n=g.n.iloc[0], cond=cond, peak_rel=g.rel.iloc[int(np.argmax(sm))],
                        peak=g.success.max(), w50=(sm >= 0.5 * sm.max()).sum() * np.diff(g.loglam).mean()
                        if sm.max() > 0 else 0))
    p = pd.DataFrame(out)
    print("\nper-instance median: peak log10(lambda*/lambda_exact), peak success, w50 (decades, coarse)")
    print(p.groupby(["n", "cond"])[["peak_rel", "peak", "w50"]].median().unstack("cond").round(2).to_string())
    qp = df[df.cond == "QPU"].set_index(["inst_id", "loglam"]).success
    print("\nRMSE of success vs QPU (all points)")
    for c in CONDS:
        s = df[df.cond == c].set_index(["inst_id", "loglam"]).success
        print(f"  {c:10s} {np.sqrt(((s - qp.reindex(s.index)) ** 2).mean()):.3f}")


if __name__ == "__main__":
    main()
