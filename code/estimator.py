"""Sample-based lambda selection vs. fixed rules, under a QPU-like sampling condition.

est     : iterative estimate of lambda_exact from samples (E* unknown), then 1.5x
grid    : budget-matched grid search (same number of reads as est), pick best feasible-optimum count
          proxy = lowest feasible objective found, ties -> more hits
wc, inst: fixed bounds (2n-1)B and max_k Delta_k
oracle  : 1.5 * lambda_exact (needs E*, not available in practice)

Every method's chosen lambda is scored with fresh reads on the same instance.

    python estimator.py --out estimator.csv --cond C3_ice02 --workers 8
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

import gonogo

ROUNDS, READS_ROUND, READS_EVAL = 4, 250, 1000
GRID = np.linspace(0.5, 2.5, ROUNDS)
MARGIN = np.log10(1.5)


def draw(inst, loglam, cond, reads, rng, sampler, n_prog=1):
    """Sample (reads split over n_prog noise draws); returns objective, violations, global-optimum hit mask."""
    n = inst["U"].shape[0]
    h, J = gonogo.qubo_to_ising(inst["U"] + 10.0 ** loglam * inst["P"])
    cfg = dict(n_prog=n_prog, reads=reads // n_prog, sweeps=1000)
    S = np.concatenate(gonogo.sample_condition(h, J, cond, cfg, rng, sampler))
    k = ((S + 1) // 2) @ (1 << np.arange(n))
    return inst["E"][k], inst["V"][k], inst["feas"][k] & (inst["E"][k] == inst["E_star"])


def estimate(inst, cond, rng, sampler):
    """Iterative lambda_exact estimate using only sampled (objective, violations) pairs."""
    loglam = np.log10(np.abs(inst["U"]).mean())  # start near the typical objective coefficient
    E_all, V_all = [], []
    trace = []
    for _ in range(ROUNDS):
        E, V, _ = draw(inst, loglam, cond, READS_ROUND, rng, sampler)
        E_all.append(E)
        V_all.append(V)
        E, V = np.concatenate(E_all), np.concatenate(V_all)
        if (V == 0).any():
            e_hat = E[V == 0].min()
            bad = (V > 0) & (E < e_hat)
            lam_hat = ((e_hat - E[bad]) / V[bad]).max() if bad.any() else 10.0 ** loglam / 1.5
            loglam = np.log10(1.5 * max(lam_hat, 1e-3))
        else:
            loglam += 0.5  # nothing feasible yet: raise the penalty
        trace.append(loglam)
    return loglam, trace


def sequential(inst, cond, rng, sampler):
    """Budget-matched sequential penalty search (in the spirit of Garcia et al. 2022): raise lambda until the
    lowest penalized-energy sample is feasible, then bisect between the last infeasible and first feasible."""
    lo = hi = None
    ll = np.log10(np.abs(inst["U"]).mean())
    for _ in range(ROUNDS):
        E, V, _ = draw(inst, ll, cond, READS_ROUND, rng, sampler)
        if V[np.argmin(E + 10.0 ** ll * V)] == 0:
            hi = ll
        else:
            lo = ll
        ll = (lo + hi) / 2 if lo is not None and hi is not None else ll + (0.5 if hi is None else -0.5)
    return hi if hi is not None else ll


def _garcia_run(inst, w, cond, rng, sampler):
    """One solver run in Garcia et al.'s sense: the lowest penalized-energy sample of READS_ROUND reads."""
    E, V, _ = draw(inst, np.log10(w), cond, READS_ROUND, rng, sampler)
    k = np.argmin(E + w * V)
    return V[k] == 0, E[k]


def garcia_standard(inst, cond, rng, sampler, iters=10):
    """Garcia, Ayodele & Moraglio (2022), Algorithm 1: w = 1, multiply by 10 until the solution is feasible."""
    w = 1.0
    for r in range(1, iters + 1):
        feasible, _ = _garcia_run(inst, w, cond, rng, sampler)
        if feasible:
            return np.log10(w), r
        w *= 10
    return np.log10(w / 10), iters  # no feasible solution: report the last weight tried


def garcia_upper_bound(inst):
    """Sum-of-coefficients bound: max f <= sum of positive terms, min f >= sum of negative terms."""
    return float(np.abs(inst["U"]).sum())


def garcia_binary(inst, cond, rng, sampler, iters=10):
    """Garcia et al. (2022), Algorithm 3: bisect [1, w_U] at round(sqrt(a b)); feasible -> b = w, else a = w.

    Returns the w of the best feasible solution found (ties -> smaller w), else w_U. Stops early when the
    rounded midpoint no longer splits the interval."""
    a, b = 1.0, max(round(garcia_upper_bound(inst)), 1.0)
    best, rounds = None, 0
    for _ in range(iters):
        w = float(round(np.sqrt(a * b)))
        if rounds and w in (a, b):
            break
        rounds += 1
        feasible, f = _garcia_run(inst, w, cond, rng, sampler)
        if feasible:
            if best is None or (f, w) < best:
                best = (f, w)
            b = w
        else:
            a = w
    return np.log10(best[1] if best else b), rounds


def grid_search(inst, cond, rng, sampler):
    best = None
    for ll in GRID:
        E, V, _ = draw(inst, ll, cond, READS_ROUND, rng, sampler)
        if (V == 0).any():
            e = E[V == 0].min()
            hits = int(((V == 0) & (E == e)).sum())
            if best is None or (e, -hits) < (best[0], -best[1]):
                best = (e, hits, ll)
    return best[2] if best else GRID[-1]


def run_instance(task):
    n, M, idx, seed, cond = task
    inst = gonogo.make_instance(n, M, seed)
    inst["V"] = violations(n, inst)
    return evaluate(inst, cond, seed, dict(inst_id=f"n{n}_M{M}_{idx}", n=n, M=M))


def evaluate(inst, cond, seed, meta):
    """Choose lambda by every method and score each choice with fresh reads.

    inst needs U, P, E, V (penalty value per state, 0 iff feasible), feas, E_star,
    lam_exact, lam_inst, and lam_wc (the problem's conventional fixed rule)."""
    from dwave.samplers import SimulatedAnnealingSampler
    sampler = SimulatedAnnealingSampler()
    rng = np.random.default_rng(seed + 11)
    est, trace = estimate(inst, cond, rng, sampler)
    seq = sequential(inst, cond, rng, sampler)
    # Garcia et al. (2022) as published; own rng so the methods above keep their earlier draws.
    g_rng = np.random.default_rng(seed + 23)
    garcia = {"g_std": garcia_standard(inst, cond, g_rng, sampler),
              "g_bin": garcia_binary(inst, cond, g_rng, sampler),
              "g_std4": garcia_standard(inst, cond, g_rng, sampler, iters=ROUNDS),
              "g_bin4": garcia_binary(inst, cond, g_rng, sampler, iters=ROUNDS)}
    # Margin ablation: est without its 1.5x margin, and seq given the same 1.5x margin.
    chosen = dict(est=est, est_nomargin=est - MARGIN, seq=seq, seq15=seq + MARGIN,
                  grid=grid_search(inst, cond, rng, sampler),
                  wc=np.log10(inst["lam_wc"]), inst=np.log10(inst["lam_inst"]),
                  oracle=np.log10(max(1.5 * inst["lam_exact"], 1e-3)),
                  **{k: v[0] for k, v in garcia.items()},
                  g_bin15=garcia["g_bin"][0] + MARGIN, g_bin4_15=garcia["g_bin4"][0] + MARGIN)
    rounds = dict(est=ROUNDS, est_nomargin=ROUNDS, seq=ROUNDS, seq15=ROUNDS, grid=ROUNDS,
                  **{k: v[1] for k, v in garcia.items()},
                  g_bin15=garcia["g_bin"][1], g_bin4_15=garcia["g_bin4"][1])
    rows = []
    for method, ll in chosen.items():
        # Common random numbers: every method is scored with the same noise draws and SA seeds.
        _, _, hit = draw(inst, ll, cond, READS_EVAL, np.random.default_rng(seed + 99), sampler, n_prog=4)
        rows.append(dict(meta, cond=cond, method=method,
                         loglam=ll, success=hit.mean(), rounds=rounds.get(method, 0),
                         log_ratio_exact=ll - np.log10(inst["lam_exact"]) if inst["lam_exact"] > 0 else np.nan,
                         est_final_vs_exact=(est - np.log10(1.5 * inst["lam_exact"]))
                         if inst["lam_exact"] > 0 else np.nan))
    return rows


def violations(n, inst):
    # P sums the forbidden-assignment indicators up to a constant; recover the per-state count.
    X = gonogo.all_states(n).astype(float)
    pen = ((X @ np.triu(inst["P"], 1)) * X).sum(1) + X @ np.diag(inst["P"])
    return np.rint(pen - pen[inst["feas"]].min()).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="estimator.csv")
    ap.add_argument("--cond", default="C3_ice02")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    cfg = gonogo.FULL
    tasks = []
    for n in cfg["ns"]:
        for rho in cfg["rhos"]:
            M = max(1, round(rho * n))
            for idx in range(cfg["n_inst"]):
                seed = int(np.random.SeedSequence([args.seed, n, M, idx]).generate_state(1)[0])
                tasks.append((n, M, idx, seed, args.cond))
    rows = []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(run_instance, tasks):
            rows += r
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    piv = df.pivot_table(index=["n", "M"], columns="method", values="success", aggfunc="mean")
    print(f"mean success with {READS_EVAL} fresh reads, cond {args.cond}")
    print(piv[["oracle", "est", "seq15", "est_nomargin", "seq", "grid", "g_std", "g_bin", "g_bin15", "g_std4", "g_bin4", "g_bin4_15", "inst", "wc"]].round(3).to_string())
    pick = df.pivot_table(index=["n", "M"], columns="method", values="log_ratio_exact", aggfunc="median")
    print("\nmedian log10(chosen / lambda_exact)")
    print(pick[["oracle", "est", "seq15", "est_nomargin", "seq", "g_std", "g_bin", "g_std4", "g_bin4"]].round(3).to_string())
    est = df[df.method == "est"]
    print("\nest: log10(chosen / (1.5 lambda_exact)) quantiles:",
          est.est_final_vs_exact.quantile([0.1, 0.5, 0.9]).round(2).tolist())


if __name__ == "__main__":
    main()
