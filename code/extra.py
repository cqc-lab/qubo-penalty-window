"""Margin sensitivity, per-instance best lambda, a cross-entropy baseline, and held-out instance families (C3).

For each instance, on top of estimator.evaluate's methods (held-out families only; the original 450 are in
estimator.csv), every extra choice is scored with the same common random numbers as estimator.evaluate:
  est_m{m}    : est's threshold estimate times margin m (m = 1.5 is est itself)
  oracle_m{m} : m * lambda_exact
  best        : best success over a fine grid (absolute log10 lambda in [-1, 3] and c * lambda_exact)
  ce_paper    : cross-entropy as in Roch et al. (ICCS 2021): 10 generations x 100 candidates x 100 reads,
                fitness = fraction of reads that hit the known optimum (needs E*)
  ce_budget   : the same with 4 generations x 10 candidates x 25 reads (= est's 1000 reads), and fitness from
                the samples alone (lowest feasible objective, ties -> more hits)

Families (margin 1.5 frozen, never tuned on them):
  orig    the 450 instances of estimator.csv
  rho075  same generator, M = 0.75 n (a density not used before), new seeds
  gauss   Gaussian real objective coefficients (sd 5), M = rho n
  sparse  integer objective on a random 3-regular graph, M = rho n
  onehot  dense integer objective, one-hot constraints over groups of 3 variables

    python extra.py --family orig --workers 30
"""
import argparse
from multiprocessing import Pool

import networkx as nx
import numpy as np
import pandas as pd

import estimator
import gonogo

COND = "C3_ice02"
MARGINS = [1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
FAMILIES = {
    "orig": dict(ns=[10, 14, 18], rhos=[0.25, 0.5, 1.0], n_inst=50),
    "rho075": dict(ns=[10, 14, 18], rhos=[0.75], n_inst=30),
    "gauss": dict(ns=[10, 14, 18], rhos=[0.5, 1.0], n_inst=20),
    "sparse": dict(ns=[14, 18], rhos=[0.5, 1.0], n_inst=20),
    "onehot": dict(ns=[12, 15, 18], rhos=[None], n_inst=20),
}


# ---------------------------------------------------------------- instances

def qubo_values(X, W):
    """x^T W x for upper-triangular W (diagonal = linear terms), for every row of X."""
    Xf = X.astype(float)
    return ((Xf @ np.triu(W, 1)) * Xf).sum(1) + Xf @ np.diag(W)


def finish(n, U, P, b_wc):
    X = gonogo.all_states(n)
    E = qubo_values(X, U)
    pen = qubo_values(X, P)
    pen = pen - pen.min()
    feas = np.isclose(pen, 0)
    E_star = E[feas].min()
    V = np.rint(pen).astype(int)
    lam_exact = 0.0
    if (~feas).any():
        lam_exact = max(0.0, float(((E_star - E[~feas]) / V[~feas]).max()))
    A = np.abs(U + U.T)
    delta = np.abs(np.diag(U)) + A.sum(1) - np.diag(A)
    return dict(U=U, P=P, E=E, V=V, feas=feas, E_star=E_star, lam_exact=lam_exact,
                lam_inst=float(delta.max()), lam_wc=float((2 * n - 1) * b_wc))


def two_sat_penalty(n, M, rng):
    """M random clauses, each kept only if the set stays satisfiable (checked by enumeration)."""
    X = gonogo.all_states(n)
    ok = np.ones(len(X), bool)
    P = np.zeros((n, n))
    placed, tries = 0, 0
    while placed < M:
        tries += 1
        if tries > 1000 * M:
            raise RuntimeError("could not place constraints")
        i, j = (int(v) for v in rng.choice(n, 2, replace=False))
        fi, fj = gonogo.FORBID[int(rng.integers(4))]
        viol = (X[:, i] == fi) & (X[:, j] == fj)
        if not (ok & ~viol).any():
            continue
        ok &= ~viol
        placed += 1
        a, b = (0.0, 1.0) if fi else (1.0, -1.0)  # literal l_f(x) = a + b x
        c, d = (0.0, 1.0) if fj else (1.0, -1.0)
        P[i, i] += b * c
        P[j, j] += a * d
        P[min(i, j), max(i, j)] += b * d
    return P


def dense_int(n, rng):
    Q = rng.integers(-gonogo.B, gonogo.B + 1, size=(n, n)).astype(float)
    return np.triu(Q + Q.T, 1) + np.diag(np.diag(Q))


def make(family, n, rho, idx):
    seed = int(np.random.SeedSequence([4242, len(family), n, idx, int(100 * (rho or 0))]).generate_state(1)[0])
    rng = np.random.default_rng(seed)
    if family == "rho075":
        inst = finish(n, dense_int(n, rng), two_sat_penalty(n, round(rho * n), rng), gonogo.B)
    elif family == "gauss":
        Q = rng.normal(0, 5.0, size=(n, n))
        U = np.triu(Q + Q.T, 1) + np.diag(np.diag(Q))
        inst = finish(n, U, two_sat_penalty(n, round(rho * n), rng), np.abs(U).max())
    elif family == "sparse":
        g = nx.random_regular_graph(3, n, seed=int(rng.integers(2**31 - 1)))
        U = np.diag(rng.integers(-gonogo.B, gonogo.B + 1, n).astype(float))
        for i, j in g.edges:
            U[min(i, j), max(i, j)] = rng.choice([v for v in range(-gonogo.B, gonogo.B + 1) if v != 0])
        inst = finish(n, U, two_sat_penalty(n, round(rho * n), rng), gonogo.B)
    elif family == "onehot":
        P = np.zeros((n, n))
        for g0 in range(0, n, 3):  # (sum_g x - 1)^2 = -sum x_i + 2 sum_{i<j} x_i x_j + 1
            for i in range(g0, g0 + 3):
                P[i, i] -= 1
                for j in range(i + 1, g0 + 3):
                    P[i, j] += 2
        inst = finish(n, dense_int(n, rng), P, gonogo.B)
    M = round(rho * n) if rho else n // 3
    return inst, seed, dict(family=family, inst_id=f"{family}_n{n}_M{M}_{idx}", n=n, M=M)


def make_orig(n, rho, idx):
    M = max(1, round(rho * n))
    seed = int(np.random.SeedSequence([2026, n, M, idx]).generate_state(1)[0])
    inst = gonogo.make_instance(n, M, seed)
    inst["V"] = estimator.violations(n, inst)
    return inst, seed, dict(family="orig", inst_id=f"n{n}_M{M}_{idx}", n=n, M=M)


# ---------------------------------------------------------------- cross-entropy baseline

def cross_entropy(inst, rng, sampler, gens, pop, reads, fitness, elite=0.1):
    """CE over log10(lambda) with a truncated normal, as in Roch et al. (single penalty weight)."""
    mu, sd = np.log10(np.abs(inst["U"]).mean()) + 1.0, 1.0  # wide start, covering the plausible range
    for _ in range(gens):
        cand = np.clip(rng.normal(mu, sd, pop), -1.0, 4.0)
        scores = []
        for ll in cand:
            E, V, hit = estimator.draw(inst, ll, COND, reads, rng, sampler)
            if fitness == "hits":
                scores.append(hit.mean())
            else:  # sample-only proxy: lowest feasible objective (lower is better), then more hits on it
                if (V == 0).any():
                    e = E[V == 0].min()
                    scores.append(-e + 1e-6 * ((V == 0) & (E == e)).sum())
                else:
                    scores.append(-np.inf)
        k = max(1, int(round(elite * pop)))
        top = cand[np.argsort(scores)[::-1][:k]]
        mu, sd = top.mean(), max(top.std(), 0.02)
    return mu


# ---------------------------------------------------------------- per instance

def run(task):
    from dwave.samplers import SimulatedAnnealingSampler
    sampler = SimulatedAnnealingSampler()
    family, n, rho, idx, do_ce = task
    inst, seed, meta = make_orig(n, rho, idx) if family == "orig" else make(family, n, rho, idx)
    rows = [] if family == "orig" else estimator.evaluate(inst, COND, seed, dict(meta))
    # est exactly as in estimator.evaluate (same rng stream), so est_m1.5 reproduces est.
    est, _ = estimator.estimate(inst, COND, np.random.default_rng(seed + 11), sampler)
    lam_hat = est - estimator.MARGIN
    lex = inst["lam_exact"]
    chosen = {f"est_m{m}": lam_hat + np.log10(m) for m in MARGINS}
    chosen.update({f"oracle_m{m}": np.log10(max(m * lex, 1e-3)) for m in MARGINS})
    if do_ce:
        crng = np.random.default_rng(seed + 31)
        chosen["ce_budget"] = cross_entropy(inst, crng, sampler, 4, 10, 25, "proxy")
        chosen["ce_paper"] = cross_entropy(inst, crng, sampler, 10, 100, 100, "hits")
    grid = list(np.round(np.arange(-1.0, 3.0001, 0.05), 3))
    if lex > 0:
        grid += list(np.log10(lex) + np.linspace(0, 1.2, 25))

    def score(ll):
        _, _, hit = estimator.draw(inst, ll, COND, estimator.READS_EVAL, np.random.default_rng(seed + 99), sampler,
                                   n_prog=4)
        return hit.mean()

    out = [dict(meta, cond=COND, method=m, loglam=ll, success=score(ll), rounds=0,
                log_ratio_exact=ll - np.log10(lex) if lex > 0 else np.nan) for m, ll in chosen.items()]
    # best: over the grid and every lambda any method chose
    curve = {ll: score(ll) for ll in grid}
    curve.update({r["loglam"]: r["success"] for r in out + rows})
    best_ll = max(curve, key=curve.get)
    out.append(dict(meta, cond=COND, method="best", loglam=best_ll, success=curve[best_ll], rounds=0,
                    log_ratio_exact=best_ll - np.log10(lex) if lex > 0 else np.nan))
    for r in rows:
        r.setdefault("family", family)
    return rows + out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=FAMILIES, required=True)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--no-ce", action="store_true")
    args = ap.parse_args()
    cfg = FAMILIES[args.family]
    tasks = [(args.family, n, rho, idx, not args.no_ce)
             for n in cfg["ns"] for rho in cfg["rhos"] for idx in range(cfg["n_inst"])]
    rows = []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(run, tasks):
            rows += r
    df = pd.DataFrame(rows)
    df.to_csv(f"extra_{args.family}.csv", index=False)
    piv = df.pivot_table(index="inst_id", columns="method", values="success")
    print(f"{args.family}: {len(piv)} instances, mean success")
    print(piv.mean().round(3).sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
