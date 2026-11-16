"""Large instances (n = 50-100) of large.py on the QPU.

  sparse : the 60 random 6-regular instances, embedded with minorminer (the interaction graph of the penalized
           model, objective and constraint couplings). est, seq and grid select lambda on the QPU as in large.py;
           every chosen lambda, the fixed bounds and the reference rule 1.5 lam_ref are then scored with EVAL reads,
           and a curve over log10(lambda / lam_ref) is measured on every other point of large.CURVE.
  dense  : 16 dense instances with n = 50 (8 per M), clique embedding; the curve only.

The reference values lam_ref and E_ref come from large.py (--ref large.csv); E_ref is lowered if the QPU finds a
better feasible value. Ocean's default chain strength. Each instance is written only when complete, and finished
instances are skipped on restart.

    python qpu_large.py --family sparse --dry-run --limit 1
    python qpu_large.py --family sparse --limit 1 --budget 5 --solver Advantage_system6
    python qpu_large.py --family sparse --budget 130 --solver Advantage_system6
    python qpu_large.py --family dense --budget 20 --solver Advantage_system6
"""
import argparse
import json
import os
import time

import minorminer
import numpy as np
import pandas as pd
from minorminer import busclique
from dwave.system import FixedEmbeddingComposite

import gonogo
import large
import qpu_run

CURVE = large.CURVE[::2]  # log10(lambda / lam_ref) from -0.5 to 1.5 in steps of 0.2
EVAL = dict(n_prog=2, reads=250)
RUN = dict(n_prog=1, reads=250)
CURVE_RUN = dict(n_prog=1, reads=250)
DENSE = dict(n=50, n_inst=8)
EMB_FILE = "qpu_large_embeddings.json"

USED_US = [0]  # QPU time of the instance in progress


def qpu_draw(inst, lam, cond, cfg, rng, sampler):
    """Drop-in for large.draw: cfg["n_prog"] gauged programmings of cfg["reads"] reads on the QPU."""
    h, J = gonogo.qubo_to_ising(inst["U"] + lam * inst["P"])
    Es, Vs = [], []
    for _ in range(cfg["n_prog"]):
        S, us, cb = qpu_run.sample_gauged(sampler, h, J, rng, DRY[0], num_reads=cfg["reads"])
        USED_US[0] += us
        CB.append(cb)
        E, V = large.evaluate_states(inst, S)
        Es.append(E)
        Vs.append(V)
    return np.concatenate(Es), np.concatenate(Vs)


DRY, CB = [False], []
large.draw = qpu_draw   # large.estimate / sequential / grid_search now sample on the QPU
large.RUN = RUN


def embedding_for(inst, key, family, n, graph, cache, seed):
    if key not in cache:
        if family == "dense":
            emb = busclique.find_clique_embedding(n, graph)
        else:
            W = np.abs(inst["U"]) + np.abs(inst["P"])  # U + lam P may cancel at one lambda but not at others
            edges = [(i, j) for i in range(n) for j in range(i + 1, n) if W[i, j] != 0]
            emb = minorminer.find_embedding(edges, graph.edges, random_seed=seed, tries=20, timeout=120)
        if not emb:
            raise RuntimeError(f"no embedding for {key}")
        cache[key] = {str(v): list(map(int, emb[v])) for v in range(n)}
        with open(EMB_FILE + (".dry" if DRY[0] else ""), "w") as f:
            json.dump(cache, f)
    return {int(v): c for v, c in cache[key].items()}


def run_instance(ref, family, child, graph, chip, cache, rng):
    n, M, idx = int(ref["n"]), int(ref["M"]), int(ref["idx"])
    inst, seed = large.make_large(family, n, M, idx)
    key = f"{chip}:{family}_{n}_{M}_{idx}" if family == "sparse" else f"{chip}:K{n}"
    emb = embedding_for(inst, key, family, n, graph, cache, seed)
    sampler = FixedEmbeddingComposite(child, emb)
    lam_ref = ref["lam_ref"]
    USED_US[0] = 0
    CB.clear()
    pool, chosen, evals = [], {}, {}
    if family == "sparse":
        est = large.estimate(inst, rng, sampler, pool)
        seq = large.sequential(inst, rng, sampler, pool)
        grid = large.grid_search(inst, rng, sampler, pool)
        m = np.log10(1.5)
        chosen = dict(est=est, est_nomargin=est - m, seq=seq, seq15=seq + m, grid=grid,
                      wc=np.log10(inst["lam_wc"]), inst=np.log10(inst["lam_inst"]), ref=np.log10(1.5 * lam_ref))
    jobs = [("method", k) for k in chosen] + [("curve", c) for c in CURVE]
    rng.shuffle(jobs)
    curve = {}
    for kind, k in jobs:
        if kind == "method":
            evals[k] = qpu_draw(inst, 10.0 ** chosen[k], None, EVAL, rng, sampler)
        else:
            curve[k] = qpu_draw(inst, lam_ref * 10.0 ** k, None, CURVE_RUN, rng, sampler)
    everything = pool + list(evals.values()) + list(curve.values())
    E_new = min((E[V == 0].min() for E, V in everything if (V == 0).any()), default=np.inf)
    E_ref = min(ref["E_ref"], E_new)
    meta = dict(family=family, n=n, M=M, idx=idx, lam_ref=lam_ref, E_ref=E_ref,
                E_ref_improved=bool(E_new < ref["E_ref"]), lam_wc=inst["lam_wc"], lam_inst=inst["lam_inst"],
                chain_len=float(np.mean([len(c) for c in emb.values()])),
                chain_max=int(max(len(c) for c in emb.values())), chip=chip)
    rows = [dict(meta, kind="method", method=k, rel=chosen[k] - np.log10(lam_ref), **large.metrics(*evals[k], E_ref))
            for k in chosen]
    rows += [dict(meta, kind="curve", method="curve", rel=c, **large.metrics(*curve[c], E_ref)) for c in CURVE]
    for r in rows:
        r.update(chain_break=float(np.mean(CB)), qpu_us_instance=USED_US[0])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=["sparse", "dense"], required=True)
    ap.add_argument("--ref", default="large.csv")
    ap.add_argument("--budget", type=float, default=130, help="max QPU access seconds for this invocation")
    ap.add_argument("--month-cap", type=float, default=560, help="max QPU seconds per calendar month (ledger)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--solver", default="Advantage_system6")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    DRY[0] = args.dry_run

    tag = "dry" if args.dry_run else "qpu"
    out = args.out or f"{tag}_large.csv"
    ledger_path = f"{tag}_ledger.json" if args.dry_run else qpu_run.LEDGER
    month = time.strftime("%Y-%m")
    refs = pd.read_csv(args.ref).drop_duplicates(["family", "n", "M", "idx"])
    refs = refs[refs.family == args.family]
    if args.family == "dense":
        refs = refs[(refs.n == DENSE["n"]) & (refs.idx < DENSE["n_inst"])]
    refs = refs.sort_values(["n", "M", "idx"]).to_dict("records")
    done = set()
    if os.path.exists(out):
        prev = pd.read_csv(out).drop_duplicates(["family", "n", "M", "idx"])
        done = set(zip(prev.family, prev.n, prev.M, prev.idx))
    refs = [r for r in refs if (args.family, r["n"], r["M"], r["idx"]) not in done]

    child, graph, chip = qpu_run.make_sampler(args.dry_run, None if args.dry_run else args.solver)
    cache = qpu_run.load_ledger(EMB_FILE + (".dry" if args.dry_run else ""))
    print(f"{chip}: {len(refs)} {args.family} instances to run, {len(done)} done; "
          f"month {month} used {qpu_run.load_ledger(ledger_path).get(month, 0) / 1e6:.1f}s", flush=True)

    used_us, ran = 0, 0
    first_est = 2.5e6 if args.family == "sparse" else 0.6e6
    for ref in refs:
        if args.limit is not None and ran >= args.limit:
            break
        next_est = used_us / ran if ran else first_est
        month_used = qpu_run.load_ledger(ledger_path).get(month, 0)
        if used_us + next_est > args.budget * 1e6 or month_used + next_est > args.month_cap * 1e6:
            print(f"stopping: budget would be exceeded (this run {used_us / 1e6:.1f}s, month {month_used / 1e6:.1f}s)")
            break
        rng = np.random.default_rng(int(np.random.SeedSequence([2026, 77, ref["n"], ref["M"], ref["idx"]])
                                        .generate_state(1)[0]))
        tag_i = f"{args.family} n={ref['n']} M={ref['M']} #{ref['idx']}"
        try:
            rows = run_instance(ref, args.family, child, graph, chip, cache, rng)
        except Exception as e:  # log the time spent, keep finished instances, stop
            qpu_run.add_to_ledger(ledger_path, month, USED_US[0])
            print(f"stopped during {tag_i} ({type(e).__name__}: {e}); instance not saved")
            break
        pd.DataFrame(rows).to_csv(out, mode="a", index=False, header=not os.path.exists(out))
        total = qpu_run.add_to_ledger(ledger_path, month, USED_US[0])
        used_us += USED_US[0]
        ran += 1
        print(f"{tag_i}: chain {rows[0]['chain_len']:.2f} (max {rows[0]['chain_max']}), "
              f"{USED_US[0] / 1e6:.2f}s; this run {used_us / 1e6:.1f}s, month {total / 1e6:.1f}s", flush=True)

    if os.path.exists(out):
        df = pd.read_csv(out)
        df = df[df.family == args.family]
        m = df[df.kind == "method"]
        if len(m):
            print("\nnear5 by method")
            print(m.pivot_table(index=["n", "M"], columns="method", values="near5").round(3).to_string())
        c = df[df.kind == "curve"]
        print("\nnear5 curve (mean) by log10(lambda / lam_ref)")
        print(c.pivot_table(index="rel", columns=["n", "M"], values="near5").round(3).to_string())


if __name__ == "__main__":
    main()
