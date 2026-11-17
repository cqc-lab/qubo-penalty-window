"""Controlled test of the temperature scale: the same embedded problem programmed at full and reduced strength.

For each main instance (n = 10 and 18, lambda_exact > 0) and each lambda = 10^rel lambda_exact, two random
spin-reversal gauges are drawn. For every gauge the embedded problem (Ocean's default chain strength) is divided
by the QPU's own auto-scale factor s_auto (extended J range, per-qubit coupling limit); the problem terms (all h
and the couplers between chains) are then multiplied by k in KS, while the chain couplers keep their k = 1
hardware value, and the result is submitted with auto_scale=False. Within a gauge the programmings differ only in
the programmed strength of the problem terms; the logical problem, the gauge, the embedding and the chain couplers
are identical. Jobs are shuffled across instances, lambda, gauge and k.

If the QPU samples at a fixed inverse temperature in programmed units, success should follow the SA master curve
in beta_eff * k * g / s_auto; if the temperature were fixed in the units of the logical problem, k would not matter.

    python qpu_strength.py --dry-run --limit 1
    python qpu_strength.py --limit 1 --budget 3 --solver Advantage_system6
    python qpu_strength.py --budget 60 --solver Advantage_system6
"""
import argparse
import json
import os
import time

import dimod
import numpy as np
import pandas as pd
from dwave.embedding import embed_bqm, unembed_sampleset
from dwave.embedding.chain_breaks import majority_vote
from dwave.embedding.chain_strength import uniform_torque_compensation

import gonogo
import qpu_run
from embedded_autoscale import autoscale

NS, RHOS, N_INST = [10, 18], [0.5, 1.0], 8
RELS = [0.18, 0.3, 0.6, 1.0]  # log10(lambda / lambda_exact)
KS = [1.0, 0.7, 0.5]
READS = 250


def make_child(dry_run, solver):
    if dry_run:
        import dwave_networkx as dnx
        from dwave.samplers import SimulatedAnnealingSampler
        g = dnx.pegasus_graph(16)
        return SimulatedAnnealingSampler(), {u: set(g[u]) for u in g}, "dry-run Pegasus P16"
    from dwave.system import DWaveSampler
    child = DWaveSampler(solver=solver)
    adj = {u: set(v) for u, v in child.adjacency.items()}
    return child, adj, child.properties["chip_id"]


def run_job(child, adj, emb, inst, n, loglam, g, k, dry_run, seed):
    h, J = gonogo.qubo_to_ising(inst["U"] + 10.0 ** loglam * inst["P"])
    hg, Jg = h * g, J * np.outer(g, g)
    bqm = dimod.BinaryQuadraticModel({i: hg[i] for i in range(n)},
                                     {(i, j): Jg[i, j] for i in range(n) for j in range(i + 1, n) if Jg[i, j]},
                                     0.0, "SPIN")
    cs = uniform_torque_compensation(bqm, emb)
    tb = embed_bqm(bqm, emb, adj, chain_strength=cs)
    s_auto, term = autoscale(tb)
    tb.scale(1.0 / s_auto)
    if k != 1.0:  # scale the problem terms only; chain couplers keep their full-strength hardware value
        chains = [set(c) for c in emb.values()]
        for v in tb.variables:
            tb.set_linear(v, k * tb.get_linear(v))
        for (u, v), b in list(tb.quadratic.items()):
            if not any(u in c and v in c for c in chains):
                tb.set_quadratic(u, v, k * b)
    assert autoscale(tb)[0] <= 1.0 + 1e-9, "programmed problem outside the hardware ranges"
    if dry_run:  # fixed inverse temperature 5 in programmed units, as the C1 emulation
        ss = child.sample(tb, num_reads=READS, num_sweeps=1000, beta_range=(0.1, 5.0), seed=seed)
        us = qpu_run.DRY_RUN_US_PER_JOB
    else:
        ss = child.sample(tb, num_reads=READS, answer_mode="raw", auto_scale=False, label="penalty-strength")
        us = int(ss.info["timing"]["qpu_access_time"])
    cb = float(np.mean(ss.record.chain_break_fraction)) if "chain_break_fraction" in ss.record.dtype.names \
        else float(np.mean([any(len({s[q] for q in c}) > 1 for c in emb.values()) for s in ss.samples()]))
    un = unembed_sampleset(ss, emb, bqm, chain_break_method=majority_vote)
    S = un.record.sample[:, np.argsort(np.asarray(list(un.variables)))] * g  # undo the gauge
    x = ((S + 1) // 2) @ (1 << np.arange(n))
    f = inst["feas"][x]
    succ = int((f & (inst["E"][x] == inst["E_star"])).sum())
    return dict(s_auto=s_auto, auto_term=term, chain_strength=cs, succ=succ, feas=int(f.sum()), reads=len(x),
                chain_break=cb, qpu_us=us)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=float, default=60, help="max QPU access seconds for this invocation")
    ap.add_argument("--month-cap", type=float, default=560, help="max QPU seconds per calendar month (ledger)")
    ap.add_argument("--limit", type=int, default=None, help="run at most this many instances")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--solver", default="Advantage_system6")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tag = "dry" if args.dry_run else "qpu"
    out = args.out or f"{tag}_strength.csv"
    ledger_path = f"{tag}_ledger.json" if args.dry_run else qpu_run.LEDGER
    month = time.strftime("%Y-%m")
    done = set(pd.read_csv(out, usecols=["inst_id"]).inst_id) if os.path.exists(out) else set()
    child, adj, chip = make_child(args.dry_run, args.solver)
    embs = json.load(open("qpu_embeddings.json"))
    print(f"{chip}: month {month} used {qpu_run.load_ledger(ledger_path).get(month, 0) / 1e6:.1f}s", flush=True)

    used_us, ran = 0, 0
    for n in NS:
        emb = {int(v): c for v, c in embs[str(n)].items()}
        for rho in RHOS:
            M = max(1, round(rho * n))
            for idx in range(N_INST):
                iid = f"n{n}_M{M}_{idx}"
                if iid in done:
                    continue
                if args.limit is not None and ran >= args.limit:
                    break
                seed = int(np.random.SeedSequence([2026, n, M, idx]).generate_state(1)[0])
                inst = gonogo.make_instance(n, M, seed)
                if inst["lam_exact"] <= 0:
                    continue
                next_est = used_us / ran if ran else len(RELS) * 2 * len(KS) * 70_000
                month_used = qpu_run.load_ledger(ledger_path).get(month, 0)
                if used_us + next_est > args.budget * 1e6 or month_used + next_est > args.month_cap * 1e6:
                    print(f"stopping before {iid}: budget would be exceeded (this run {used_us / 1e6:.1f}s)")
                    break
                rng = np.random.default_rng(seed + 53)
                gauges = {(rel, gi): rng.choice([-1, 1], n) for rel in RELS for gi in range(2)}
                jobs = [(rel, gi, k) for rel in RELS for gi in range(2) for k in KS]
                rng.shuffle(jobs)
                rows, inst_us = [], 0
                for rel, gi, k in jobs:
                    loglam = float(np.log10(inst["lam_exact"]) + rel)
                    r = run_job(child, adj, emb, inst, n, loglam, gauges[rel, gi], k, args.dry_run,
                                int(rng.integers(2**31 - 1)))
                    inst_us += r["qpu_us"]
                    rows.append(dict(inst_id=iid, n=n, M=M, rel=rel, loglam=loglam, gauge=gi, k=k, chip=chip,
                                     lam_exact=inst["lam_exact"], **r))
                pd.DataFrame(rows).to_csv(out, mode="a", index=False, header=not os.path.exists(out))
                total = qpu_run.add_to_ledger(ledger_path, month, inst_us)
                used_us += inst_us
                ran += 1
                print(f"{iid}: {inst_us / 1e6:.2f}s; this run {used_us / 1e6:.1f}s, month {total / 1e6:.1f}s",
                      flush=True)

    if os.path.exists(out):
        df = pd.read_csv(out)
        df["success"] = df.succ / df.reads
        print("\nmean success by n, rel and k")
        print(df.pivot_table(index=["n", "rel"], columns="k", values="success").round(3).to_string())
        print("\nmean chain-break fraction by k")
        print(df.groupby("k").chain_break.mean().round(4).to_string())


if __name__ == "__main__":
    main()
