"""Estimator check on instances too large for brute force (n = 50-100).

Reference values come from a strong classical solver instead of enumeration:
  E_ref   : best feasible objective seen anywhere (heavy SA in the double-precision, auto-beta setting)
  lam_ref : bisection on lambda for "the penalized ground state (found by heavy SA) is feasible";
            computed twice with different seeds to report how reproducible it is.
Methods are then compared under the QPU-like condition, with near-optimal hit rates as well as exact hits,
and a curve over lambda / lam_ref checks whether the optimum still sits near 1.5 * lam_ref.

    python large.py --out large.csv --workers 30
"""
import argparse
from multiprocessing import Pool

import networkx as nx
import numpy as np
import pandas as pd

import gonogo

SIZES = [50, 75, 100]
FAMILIES = {"dense": None, "sparse": 6}  # sparse = random 6-regular interaction graph
RHOS = [0.5, 1.0]
N_INST = 10
COND = "C3_ice02"
REF = dict(n_prog=1, reads=100, sweeps=3000)
RUN = dict(n_prog=1, reads=250, sweeps=1000)
EVAL = dict(n_prog=4, reads=250, sweeps=1000)   # separate noise draw per programming
CURVE = np.round(np.arange(-0.5, 1.5 + 1e-9, 0.1), 2)  # log10(lambda / lam_ref)
CURVE_RUN = dict(n_prog=2, reads=250, sweeps=1000)
ROUNDS = 4
NEAR = 0.01  # near-optimal: within 1% of |E_ref|


# ---------------------------------------------------------------- instances

def lit(i, val):
    return 2 * i if val else 2 * i + 1


def two_sat_ok(n, cons):
    """Satisfiability of the clauses 'not (x_i = fi and x_j = fj)' via the implication graph."""
    G = nx.DiGraph()
    G.add_nodes_from(range(2 * n))
    for i, j, t in cons:
        fi, fj = gonogo.FORBID[t]
        a, b = lit(i, 1 - fi), lit(j, 1 - fj)  # clause a or b
        G.add_edge(a ^ 1, b)
        G.add_edge(b ^ 1, a)
    comp = {}
    for c, nodes in enumerate(nx.strongly_connected_components(G)):
        for v in nodes:
            comp[v] = c
    return all(comp[2 * i] != comp[2 * i + 1] for i in range(n))


def make_large(family, n, M, idx, seed_base=2026):
    seed = int(np.random.SeedSequence([seed_base, 99, n, M, idx, len(family)]).generate_state(1)[0])
    rng = np.random.default_rng(seed)
    B = gonogo.B
    if FAMILIES[family] is None:
        Q = rng.integers(-B, B + 1, size=(n, n)).astype(float)
        U = np.triu(Q + Q.T, 1) + np.diag(np.diag(Q))
    else:
        g = nx.random_regular_graph(FAMILIES[family], n, seed=int(rng.integers(2**31 - 1)))
        U = np.diag(rng.integers(-B, B + 1, n).astype(float))
        for i, j in g.edges:
            U[min(i, j), max(i, j)] = rng.choice([v for v in range(-B, B + 1) if v != 0])
    cons = []
    while len(cons) < M:
        i, j = (int(v) for v in rng.choice(n, 2, replace=False))
        c = (i, j, int(rng.integers(4)))
        if two_sat_ok(n, cons + [c]):
            cons.append(c)
    P = np.zeros((n, n))
    for i, j, t in cons:
        fi, fj = gonogo.FORBID[t]
        a, b = (0.0, 1.0) if fi else (1.0, -1.0)
        c, d = (0.0, 1.0) if fj else (1.0, -1.0)
        P[i, i] += b * c
        P[j, j] += a * d
        P[min(i, j), max(i, j)] += b * d
    Us = U + U.T
    offdiag = np.abs(Us) - np.diag(np.abs(np.diag(Us)))
    delta = np.abs(np.diag(U)) + offdiag.sum(1)
    maxdeg = int((offdiag > 0).sum(1).max())
    arr = np.array(cons)
    forb = np.array([gonogo.FORBID[t] for t in arr[:, 2]])
    return dict(U=U, P=P, I=arr[:, 0], J=arr[:, 1], FI=forb[:, 0], FJ=forb[:, 1],
                lam_inst=float(delta.max()), lam_wc=float((2 * maxdeg + 1) * B)), seed


def evaluate_states(inst, S):
    X = ((S + 1) // 2).astype(np.int8)
    Xf = X.astype(float)
    E = ((Xf @ inst["U"]) * Xf).sum(1)
    V = ((X[:, inst["I"]] == inst["FI"]) & (X[:, inst["J"]] == inst["FJ"])).sum(1)
    return E, V


def draw(inst, lam, cond, cfg, rng, sampler):
    h, J = gonogo.qubo_to_ising(inst["U"] + lam * inst["P"])
    S = np.concatenate(gonogo.sample_condition(h, J, cond, cfg, rng, sampler))
    return evaluate_states(inst, S)


# ---------------------------------------------------------------- reference

def reference(inst, rng, sampler, pool):
    """Bisection for the smallest lambda whose heavy-SA penalized ground state is feasible."""
    def ground_feasible(lam):
        E, V = draw(inst, lam, "C0_raw", REF, rng, sampler)
        pool.append((E, V))
        return V[np.argmin(E + lam * V)] == 0

    lo, hi = 1e-2, 2 * inst["lam_wc"]
    while not ground_feasible(hi):
        hi *= 4
    if ground_feasible(lo):
        return lo
    for _ in range(12):
        mid = np.sqrt(lo * hi)
        if ground_feasible(mid):
            hi = mid
        else:
            lo = mid
    return hi


# ---------------------------------------------------------------- methods

def start_loglam(inst):
    nz = np.abs(inst["U"][inst["U"] != 0])
    return np.log10(nz.mean())


def estimate(inst, rng, sampler, pool):
    loglam = start_loglam(inst)
    Es, Vs = [], []
    for _ in range(ROUNDS):
        E, V = draw(inst, 10.0 ** loglam, COND, RUN, rng, sampler)
        pool.append((E, V))
        Es.append(E)
        Vs.append(V)
        E, V = np.concatenate(Es), np.concatenate(Vs)
        if (V == 0).any():
            e_hat = E[V == 0].min()
            bad = (V > 0) & (E < e_hat)
            lam_hat = ((e_hat - E[bad]) / V[bad]).max() if bad.any() else 10.0 ** loglam / 1.5
            loglam = np.log10(1.5 * max(lam_hat, 1e-3))
        else:
            loglam += 0.5
    return loglam


def sequential(inst, rng, sampler, pool):
    """Budget-matched sequential penalty search: raise lambda until the lowest penalized-energy sample is
    feasible, then bisect between the last infeasible and first feasible lambda."""
    lo = hi = None
    ll = start_loglam(inst)
    for _ in range(ROUNDS):
        E, V = draw(inst, 10.0 ** ll, COND, RUN, rng, sampler)
        pool.append((E, V))
        if V[np.argmin(E + 10.0 ** ll * V)] == 0:
            hi = ll
        else:
            lo = ll
        ll = (lo + hi) / 2 if lo is not None and hi is not None else ll + (0.5 if hi is None else -0.5)
    return hi if hi is not None else ll


def grid_search(inst, rng, sampler, pool):
    best = None
    for ll in np.linspace(start_loglam(inst), np.log10(inst["lam_wc"]), ROUNDS):
        E, V = draw(inst, 10.0 ** ll, COND, RUN, rng, sampler)
        pool.append((E, V))
        if (V == 0).any():
            e = E[V == 0].min()
            hits = int(((V == 0) & (E == e)).sum())
            if best is None or (e, -hits) < (best[0], -best[1]):
                best = (e, hits, ll)
    return best[2] if best else np.log10(inst["lam_wc"])


def metrics(E, V, E_ref):
    f = V == 0
    tol = NEAR * abs(E_ref)
    resid = (E[f] - E_ref) / abs(E_ref) if f.any() else np.array([np.nan])
    return dict(feas=f.mean(), hit=(f & (E <= E_ref)).mean(), near=(f & (E <= E_ref + tol)).mean(),
                near5=(f & (E <= E_ref + 5 * tol)).mean(),
                resid_mean=float(np.mean(resid)), resid_best=float(np.min(resid)))


def run_instance(task):
    from dwave.samplers import SimulatedAnnealingSampler
    family, n, M, idx = task
    inst, seed = make_large(family, n, M, idx)
    sampler = SimulatedAnnealingSampler()
    rng = np.random.default_rng(seed + 3)
    pool = []
    lam_ref = reference(inst, rng, sampler, pool)
    lam_ref2 = reference(inst, np.random.default_rng(seed + 4), sampler, pool)
    heavy = list(pool)

    est = estimate(inst, rng, sampler, pool)
    seq = sequential(inst, rng, sampler, pool)
    margin = np.log10(1.5)
    # Margin ablation: est without its 1.5x margin, and seq given the same 1.5x margin.
    chosen = dict(est=est, est_nomargin=est - margin, seq=seq, seq15=seq + margin,
                  grid=grid_search(inst, rng, sampler, pool),
                  wc=np.log10(inst["lam_wc"]), inst=np.log10(inst["lam_inst"]),
                  ref=np.log10(1.5 * lam_ref))
    # Common random numbers: every method is scored with the same noise draws and SA seeds.
    evals = {m: draw(inst, 10.0 ** ll, COND, EVAL, np.random.default_rng(seed + 99), sampler)
             for m, ll in chosen.items()}
    curve = {c: draw(inst, lam_ref * 10.0 ** c, COND, CURVE_RUN, rng, sampler) for c in CURVE}

    # E_ref: best feasible objective seen in any sample of this instance.
    everything = pool + list(evals.values()) + list(curve.values())
    E_ref = min(E[V == 0].min() for E, V in everything if (V == 0).any())
    E_ref_heavy = min(E[V == 0].min() for E, V in heavy if (V == 0).any())

    meta = dict(family=family, n=n, M=M, idx=idx, lam_ref=lam_ref, lam_ref2=lam_ref2,
                E_ref=E_ref, E_ref_heavy=E_ref_heavy, lam_wc=inst["lam_wc"], lam_inst=inst["lam_inst"])
    rows = [dict(meta, kind="method", method=m, rel=chosen[m] - np.log10(lam_ref), **metrics(*evals[m], E_ref))
            for m in chosen]
    rows += [dict(meta, kind="curve", method="curve", rel=c, **metrics(*curve[c], E_ref)) for c in CURVE]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="large.csv")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    tasks = [(f, n, max(1, round(rho * n)), i) for f in FAMILIES for n in SIZES for rho in RHOS
             for i in range(N_INST)]
    if args.quick:
        tasks = [("dense", 50, 25, 0), ("sparse", 100, 100, 0)]
    rows = []
    with Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(run_instance, tasks), 1):
            rows += r
            print(f"[{k}/{len(tasks)}] {r[0]['family']} n={r[0]['n']} M={r[0]['M']} #{r[0]['idx']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    report(df)


def report(df):
    pd.set_option("display.width", 200)
    inst = df.drop_duplicates(["family", "n", "M", "idx"])
    agree = np.abs(np.log10(inst.lam_ref) - np.log10(inst.lam_ref2))
    print(f"\nlam_ref reproducibility: median |dlog10| = {agree.median():.3f}, "
          f"90% = {agree.quantile(0.9):.3f}; E_ref found by heavy SA in "
          f"{(inst.E_ref_heavy == inst.E_ref).mean():.0%} of instances")
    m = df[df.kind == "method"]
    for metric in ["near", "near5", "hit", "feas"]:
        piv = m.pivot_table(index=["family", "n", "M"], columns="method", values=metric, aggfunc="mean")
        print(f"\n{metric} (mean over instances, {COND})")
        print(piv[["ref", "est", "seq15", "est_nomargin", "seq", "grid", "inst", "wc"]].round(3).to_string())
    pick = m.pivot_table(index=["family", "n", "M"], columns="method", values="rel", aggfunc="median")
    print("\nmedian log10(chosen / lam_ref)")
    print(pick[["ref", "est", "seq15", "est_nomargin", "seq"]].round(3).to_string())
    est = m[m.method == "est"]
    print("\nest log10(lam_est / lam_ref) quantiles 10/50/90:",
          est.rel.quantile([0.1, 0.5, 0.9]).round(2).tolist(), "(1.5x = 0.18)")
    c = df[df.kind == "curve"].groupby(["family", "n", "M", "rel"])[["near", "near5", "feas"]].mean().reset_index()
    for metric in ["near", "near5"]:
        peak = c.loc[c.groupby(["family", "n", "M"])[metric].idxmax()]
        print(f"\ncurve peak of {metric} rate, log10(lambda / lam_ref):")
        print(peak.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
