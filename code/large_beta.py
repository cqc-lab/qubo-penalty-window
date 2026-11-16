"""Large instances (n = 50-100) re-run at effective temperatures calibrated to the hardware.

large.py emulates the QPU at beta = 5 (condition C3). The hardware fits give beta_eff of 1.9-3.1 for chain lengths
2-3.9 and roughly beta_eff * chain length = 6-7.5 (paper, Table 5). Clique embeddings of K50, K75 and K100 on
Advantage_system4 need chains of mean length 5.8, 7.8 and 9.8; minorminer embeddings of the random 6-regular
instances need 2.3-3.4. We therefore re-run every method at beta in BETAS (sigma = 0, as in the direct fit),
reusing the reference values lam_ref and E_ref of large.py, so only the QPU-like condition changes.
Instances, seeds and methods are those of large.py.

    python large_beta.py --ref large.csv --out large_beta.csv --workers 30
"""
import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd

import gonogo
import large

BETAS = [2.5, 2.0, 1.5, 1.0, 0.75]


def cond_name(beta):
    return f"H_b{beta:g}"


for _b in BETAS:
    gonogo.CONDITIONS[cond_name(_b)] = (True, (0.1, _b), 0.0)


def run_instance(task):
    from dwave.samplers import SimulatedAnnealingSampler
    beta, ref = task
    cond = cond_name(beta)
    large.COND = cond  # estimate / sequential / grid_search read the module-level condition
    family, n, M, idx = ref["family"], ref["n"], ref["M"], ref["idx"]
    inst, seed = large.make_large(family, n, M, idx)
    sampler = SimulatedAnnealingSampler()
    rng = np.random.default_rng(seed + 3)
    lam_ref = ref["lam_ref"]
    pool = []
    est = large.estimate(inst, rng, sampler, pool)
    seq = large.sequential(inst, rng, sampler, pool)
    margin = np.log10(1.5)
    chosen = dict(est=est, est_nomargin=est - margin, seq=seq, seq15=seq + margin,
                  grid=large.grid_search(inst, rng, sampler, pool),
                  wc=np.log10(inst["lam_wc"]), inst=np.log10(inst["lam_inst"]),
                  ref=np.log10(1.5 * lam_ref))
    evals = {m: large.draw(inst, 10.0 ** ll, cond, large.EVAL, np.random.default_rng(seed + 99), sampler)
             for m, ll in chosen.items()}
    curve = {c: large.draw(inst, lam_ref * 10.0 ** c, cond, large.CURVE_RUN, rng, sampler) for c in large.CURVE}
    everything = pool + list(evals.values()) + list(curve.values())
    E_new = min((E[V == 0].min() for E, V in everything if (V == 0).any()), default=np.inf)
    E_ref = min(ref["E_ref"], E_new)
    meta = dict(beta=beta, family=family, n=n, M=M, idx=idx, lam_ref=lam_ref, E_ref=E_ref,
                E_ref_improved=bool(E_new < ref["E_ref"]), lam_wc=inst["lam_wc"], lam_inst=inst["lam_inst"])
    rows = [dict(meta, kind="method", method=m, rel=chosen[m] - np.log10(lam_ref),
                 **large.metrics(*evals[m], E_ref)) for m in chosen]
    rows += [dict(meta, kind="curve", method="curve", rel=c, **large.metrics(*curve[c], E_ref))
             for c in large.CURVE]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="large.csv", help="output of large.py (reference values)")
    ap.add_argument("--out", default="large_beta.csv")
    ap.add_argument("--workers", type=int, default=30)
    args = ap.parse_args()
    refs = pd.read_csv(args.ref).drop_duplicates(["family", "n", "M", "idx"])
    refs = refs[["family", "n", "M", "idx", "lam_ref", "E_ref"]].to_dict("records")
    tasks = [(b, r) for b in BETAS for r in refs]
    rows = []
    with Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(run_instance, tasks), 1):
            rows += r
            if k % 20 == 0:
                print(f"[{k}/{len(tasks)}]", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    m = df[df.kind == "method"]
    for metric in ["near5", "hit", "feas"]:
        piv = m.pivot_table(index=["beta", "family", "n", "M"], columns="method", values=metric, aggfunc="mean")
        print(f"\n{metric}")
        print(piv[["ref", "est", "seq15", "est_nomargin", "seq", "grid", "inst", "wc"]].round(3).to_string())
    print("\nest log10(lam_est / lam_ref) quantiles 10/50/90 by beta:")
    print(m[m.method == "est"].groupby("beta").rel.quantile([0.1, 0.5, 0.9]).unstack().round(2).to_string())


if __name__ == "__main__":
    main()
