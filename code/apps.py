"""Application problems with constraint structures unlike 2-SAT, run through the same pipeline.

tsp      : 5 cities, city 0 fixed first -> 4x4 = 16 variables, 8 one-hot equality penalties
knapsack : m items + binary slack bits, one inequality as a squared equality penalty

The conventional fixed rule (column lam_wc) is max distance for TSP and max item value for knapsack.

    python apps.py sweep --out apps_sweep.csv      # gonogo.py-compatible rows, all conditions
    python apps.py est --out apps_est.csv          # estimator.py methods under C3_ice02
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

import estimator
import gonogo

LOGLAMS = np.round(np.arange(-2.0, 3.0 + 1e-9, 0.125), 3)
SWEEP = dict(n_prog=4, reads=125, sweeps=1000)
TASKS = [("tsp", 5, i) for i in range(40)] + [("knapsack", 10, i) for i in range(20)] + \
        [("knapsack", 12, i) for i in range(20)]


def add_square(U, idx, coef, rhs):
    """Add (sum_k coef_k z_idx_k - rhs)^2 to upper-triangular QUBO U, constant dropped."""
    for a, (i, ci) in enumerate(zip(idx, coef)):
        U[i, i] += ci * ci - 2 * rhs * ci
        for j, cj in zip(idx[a + 1:], coef[a + 1:]):
            U[min(i, j), max(i, j)] += 2 * ci * cj


def tsp(n_cities, rng):
    pts = rng.integers(0, 11, size=(n_cities, 2))
    d = np.maximum(1, np.rint(np.linalg.norm(pts[:, None] - pts[None], axis=2))).astype(float)
    k = n_cities - 1  # cities 1..k at positions 1..k
    var = lambda c, t: (c - 1) * k + (t - 1)
    n = k * k
    U = np.zeros((n, n))
    for c in range(1, n_cities):
        U[var(c, 1), var(c, 1)] += d[0, c]
        U[var(c, k), var(c, k)] += d[c, 0]
        for c2 in range(1, n_cities):
            if c2 != c:
                for t in range(1, k):
                    i, j = var(c, t), var(c2, t + 1)
                    U[min(i, j), max(i, j)] += d[c, c2]
    P = np.zeros((n, n))
    for c in range(1, n_cities):
        add_square(P, [var(c, t) for t in range(1, k + 1)], [1.0] * k, 1.0)
    for t in range(1, k + 1):
        add_square(P, [var(c, t) for c in range(1, n_cities)], [1.0] * k, 1.0)
    return U, P, float(d.max()), 2 * k


def knapsack(m, rng):
    w = rng.integers(1, 11, m).astype(float)
    v = rng.integers(1, 11, m).astype(float)
    C = float(w.sum() // 2)
    slack, total = [], 0.0
    while total < C:
        c = min(2.0 ** len(slack), C - total)
        slack.append(c)
        total += c
    n = m + len(slack)
    U = np.zeros((n, n))
    U[np.arange(m), np.arange(m)] = -v
    P = np.zeros((n, n))
    add_square(P, list(range(n)), list(w) + slack, C)
    return U, P, float(v.max()), 1


def build(problem, size, idx, seed_base=2026):
    seed = int(np.random.SeedSequence([seed_base, 7, size, idx, len(problem)]).generate_state(1)[0])
    rng = np.random.default_rng(seed)
    U, P, lam_conv, n_cons = (tsp if problem == "tsp" else knapsack)(size, rng)
    n = U.shape[0]
    X = gonogo.all_states(n).astype(float)
    E = ((X @ U) * X).sum(1)
    pen = ((X @ P) * X).sum(1)
    V = pen - pen.min()
    feas = V < 1e-9
    V[feas] = 0
    E_star = E[feas].min()
    lam_exact = max(0.0, float(((E_star - E[~feas]) / V[~feas]).max()))
    Qs = U + U.T
    delta = np.abs(np.diag(U)) + np.abs(Qs).sum(1) - 2 * np.abs(np.diag(U))
    inst = dict(U=U, P=P, E=E, V=V, feas=feas, E_star=E_star, lam_exact=lam_exact,
                lam_inst=float(max(delta.max(), 1e-3)), lam_wc=lam_conv)
    meta = dict(inst_id=f"{problem}{size}_{idx}", n=n, M=n_cons, problem=problem)
    return inst, meta, seed


def sweep_one(task):
    from dwave.samplers import SimulatedAnnealingSampler
    inst, meta, seed = build(*task)
    sampler = SimulatedAnnealingSampler()
    rng = np.random.default_rng(seed + 1)
    n = inst["U"].shape[0]
    weights = 1 << np.arange(n)
    rows = []
    for loglam in LOGLAMS:
        h, J = gonogo.qubo_to_ising(inst["U"] + 10.0 ** loglam * inst["P"])
        for cond in gonogo.CONDITIONS:
            succ, feas = [0, 0], [0, 0]
            for p, S in enumerate(gonogo.sample_condition(h, J, cond, SWEEP, rng, sampler)):
                k = ((S + 1) // 2) @ weights
                f = inst["feas"][k]
                feas[p % 2] += int(f.sum())
                succ[p % 2] += int((f & (inst["E"][k] == inst["E_star"])).sum())
            half = SWEEP["reads"] * SWEEP["n_prog"] // 2
            rows.append(dict(meta, cond=cond, loglam=loglam, succ_a=succ[0], succ_b=succ[1],
                             feas_a=feas[0], feas_b=feas[1], reads_a=half, reads_b=half,
                             lam_exact=inst["lam_exact"], lam_inst=inst["lam_inst"], lam_wc=inst["lam_wc"]))
    return rows


def est_one(task):
    inst, meta, seed = build(*task)
    return estimator.evaluate(inst, "C3_ice02", seed, meta)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["sweep", "est"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    fn = sweep_one if args.cmd == "sweep" else est_one
    rows = []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(fn, TASKS):
            rows += r
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    if args.cmd == "est":
        piv = df.pivot_table(index="problem", columns="method", values="success", aggfunc="mean")
        print(piv[["oracle", "est", "grid", "inst", "wc"]].round(3).to_string())


if __name__ == "__main__":
    main()
