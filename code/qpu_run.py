"""Run the gonogo.py sweep instances on a D-Wave QPU, with a hard budget on QPU access time.

Instances are regenerated from the same seeds as gonogo.py, so QPU rows pair with the SA rows.
Output has the same schema as gonogo.py (cond = "QPU"), so `gonogo.py analyze` works on it.

    python qpu_run.py --phase pilot --dry-run          # SA on a Pegasus-shaped graph, no QPU time used
    python qpu_run.py --phase pilot --budget 30        # real QPU; needs DWAVE_API_TOKEN
    python qpu_run.py --phase main --budget 300
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import dimod
from minorminer import busclique
from dwave.system import FixedEmbeddingComposite

import gonogo

PHASES = {
    "pilot": dict(ns=[10, 14], rhos=[0.5], n_inst=5, loglams=np.round(np.linspace(0.5, 2.25, 12), 3)),
    "main": dict(ns=[10, 14, 18], rhos=[0.5, 1.0], n_inst=8, loglams=np.round(np.linspace(0.5, 2.5, 15), 3)),
    # Larger n on the same lambda grid: chains stay at length 3, so this separates n from chain length.
    "bign": dict(ns=[20, 22], rhos=[0.5, 1.0], n_inst=8, loglams=np.round(np.linspace(0.5, 2.5, 15), 3)),
    # The main n=10 and n=18 instances again, with chains lengthened to 3 and 4 by taking the first n chains
    # of a larger clique embedding (emb_size). Pairs with "main" instance by instance.
    "chain": dict(ns=[10, 18], rhos=[0.5, 1.0], n_inst=8, loglams=np.round(np.linspace(0.5, 2.5, 15), 3),
                  emb_size={10: 14, 18: 30}),
}
READS_PER_HALF = 250  # two programmings per (instance, lambda), each with its own spin-reversal gauge
LEDGER = "qpu_ledger.json"
DRY_RUN_US_PER_JOB = 60_000


def load_ledger(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def add_to_ledger(path, month, us):
    ledger = load_ledger(path)
    ledger[month] = ledger.get(month, 0) + us
    with open(path, "w") as f:
        json.dump(ledger, f, indent=1)
    return ledger[month]


def make_sampler(dry_run, solver):
    if dry_run:
        import dwave_networkx as dnx
        from dwave.samplers import SimulatedAnnealingSampler
        g = dnx.pegasus_graph(16)
        child = dimod.StructureComposite(SimulatedAnnealingSampler(), list(g.nodes), list(g.edges))
        return child, g, "dry-run Pegasus P16"
    from dwave.system import DWaveSampler
    # Pegasus by default: its h/J ranges ([-4,4], [-1,1]) are the ones the SA emulation assumed.
    child = DWaveSampler(solver=solver or dict(topology__type="pegasus"))
    return child, child.to_networkx_graph(), child.properties["chip_id"]


def clique_embedding(n, graph, cache, size=None):
    """Clique embedding of K_n; with size > n, the first n chains of a K_size embedding (longer chains)."""
    size = size or n
    key = str(n) if size == n else f"{n}of{size}"
    if key not in cache:
        emb = busclique.find_clique_embedding(size, graph)
        if not emb:
            raise RuntimeError(f"no clique embedding for size={size}")
        cache[key] = {str(v): list(map(int, emb[v])) for v in range(n)}
    return {int(v): c for v, c in cache[key].items()}


def sample_gauged(sampler, h, J, rng, dry_run, num_reads=READS_PER_HALF):
    """One programming with a random spin-reversal gauge. Returns (spins in logical order, qpu_us, chain_break)."""
    n = len(h)
    g = rng.choice([-1, 1], n)
    hg = h * g
    Jg = J * np.outer(g, g)
    bqm = dimod.BinaryQuadraticModel(hg, Jg, 0.0, "SPIN")
    if dry_run:
        # Emulate QPU auto-scale and fixed temperature (gonogo C1) on the embedded problem.
        ss = sampler.sample(bqm, num_reads=num_reads, num_sweeps=1000,
                            beta_range=(0.1, 5.0 / max(np.abs(hg).max() / 4, np.abs(Jg).max())),
                            chain_strength=None, seed=int(rng.integers(2**31 - 1)))
        us = DRY_RUN_US_PER_JOB
    else:
        ss = sampler.sample(bqm, num_reads=num_reads, answer_mode="raw",
                            label="penalty-window")
        us = int(ss.info["timing"]["qpu_access_time"])
    order = np.argsort(np.asarray(list(ss.variables)))
    S = ss.record.sample[:, order] * g  # undo the gauge
    cb = float(np.mean(ss.record.chain_break_fraction)) if "chain_break_fraction" in ss.record.dtype.names else np.nan
    return S, us, cb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=PHASES, required=True)
    ap.add_argument("--budget", type=float, default=30, help="max QPU access seconds for this invocation")
    ap.add_argument("--month-cap", type=float, default=560, help="max QPU seconds per calendar month (ledger)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    ap.add_argument("--solver", default=None, help="QPU solver name (default: any Pegasus Advantage)")
    ap.add_argument("--emb-file", default=None, help="embedding cache (default: <tag>_embeddings.json)")
    args = ap.parse_args()

    tag = "dry" if args.dry_run else "qpu"
    out = args.out or f"{tag}_{args.phase}.csv"
    ledger_path = f"{tag}_ledger.json" if args.dry_run else LEDGER
    month = time.strftime("%Y-%m")
    cfg = PHASES[args.phase]

    done = set()
    if os.path.exists(out):
        prev = pd.read_csv(out, usecols=["inst_id", "loglam"])
        done = set(zip(prev.inst_id, prev.loglam.round(3)))

    jobs = []
    for n in cfg["ns"]:
        for rho in cfg["rhos"]:
            M = max(1, round(rho * n))
            for idx in range(cfg["n_inst"]):
                seed = int(np.random.SeedSequence([args.seed, n, M, idx]).generate_state(1)[0])
                for loglam in cfg["loglams"]:
                    if (f"n{n}_M{M}_{idx}", float(loglam)) not in done:
                        jobs.append((n, M, idx, seed, float(loglam)))
    rng = np.random.default_rng(args.seed + 7)
    rng.shuffle(jobs)  # interleave lambdas and instances so hardware drift does not align with lambda

    child, graph, chip = make_sampler(args.dry_run, args.solver)
    emb_path = args.emb_file or f"{tag}_embeddings.json"
    emb_cache = load_ledger(emb_path)
    print(f"{chip}: {len(jobs)} jobs to run, {len(done)} already done; "
          f"month {month} used {load_ledger(ledger_path).get(month, 0) / 1e6:.1f}s")

    used_us, insts, samplers = 0, {}, {}
    for k, (n, M, idx, seed, loglam) in enumerate(jobs, 1):
        month_used = load_ledger(ledger_path).get(month, 0)
        next_est = (used_us / (2 * (k - 1))) * 2 if k > 1 else 2 * 100_000
        if used_us + next_est > args.budget * 1e6 or month_used + next_est > args.month_cap * 1e6:
            print(f"stopping before job {k}: budget would be exceeded "
                  f"(this run {used_us / 1e6:.1f}s, month {month_used / 1e6:.1f}s)")
            break
        key = (n, M, idx)
        if key not in insts:
            # Keep every instance: jobs are shuffled, and rebuilding an n=22 instance takes seconds.
            insts[key] = {k: v for k, v in gonogo.make_instance(n, M, seed).items() if k in ("E", "feas", "E_star",
                          "lam_exact", "lam_inst", "lam_wc", "U", "P")}
        inst = insts[key]
        if n not in samplers:
            emb = clique_embedding(n, graph, emb_cache, cfg.get("emb_size", {}).get(n))
            print(f"n={n}: chain length {np.mean([len(c) for c in emb.values()]):.1f}", flush=True)
            samplers[n] = FixedEmbeddingComposite(child, emb)
            with open(emb_path, "w") as f:
                json.dump(emb_cache, f)

        h, J = gonogo.qubo_to_ising(inst["U"] + 10.0 ** loglam * inst["P"])
        weights = 1 << np.arange(n)
        row = dict(inst_id=f"n{n}_M{M}_{idx}", n=n, M=M, cond="QPU", loglam=loglam, phase=args.phase,
                   chain_len=float(np.mean([len(c) for c in samplers[n].embedding.values()])),
                   lam_exact=inst["lam_exact"], lam_inst=inst["lam_inst"], lam_wc=inst["lam_wc"])
        job_us, cbs = 0, []
        for half in "ab":
            S, us, cb = sample_gauged(samplers[n], h, J, rng, args.dry_run)
            x = ((S + 1) // 2) @ weights
            f = inst["feas"][x]
            row[f"succ_{half}"] = int((f & (inst["E"][x] == inst["E_star"])).sum())
            row[f"feas_{half}"] = int(f.sum())
            row[f"reads_{half}"] = len(x)
            job_us += us
            cbs.append(cb)
        row.update(chain_break=float(np.mean(cbs)), qpu_us=job_us)
        used_us += job_us
        total = add_to_ledger(ledger_path, month, job_us)
        pd.DataFrame([row]).to_csv(out, mode="a", index=False, header=not os.path.exists(out))
        if k % 10 == 0 or k == len(jobs):
            print(f"[{k}/{len(jobs)}] this run {used_us / 1e6:.1f}s, month {total / 1e6:.1f}s, "
                  f"last job {job_us / 1e3:.0f}ms, chain break {row['chain_break']:.3f}", flush=True)


if __name__ == "__main__":
    main()
