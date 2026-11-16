"""Chain length vs. beta_eff with both embeddings interleaved in one session, paired within each instance.

The chain run of qpu_run.py (phase "chain") lengthened the chains in a separate session, so its comparison with
the main run is not insulated from session-to-session drift. Here every (lambda, embedding) point of an instance
is run back to back in shuffled order:

  short : the clique embedding of K_n used in the main run (chain length 2 for n = 10, 3 for n = 18)
  long  : the first n chains of a K_14 / K_30 embedding, as in the chain run (chain length 3 and 3.9)

Instances and the lambda grid are those of the main run (same seeds), so the beta_fine.py emulation grid applies
unchanged. Ocean's default chain strength, as in the main run. Each instance is written only when complete, so a
run stopped early still leaves complete pairs; finished instances are skipped on restart.

    python qpu_interleave.py --dry-run --limit 1
    python qpu_interleave.py --limit 1 --budget 5 --solver Advantage_system4     # one instance
    python qpu_interleave.py --budget 80 --solver Advantage_system4
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from dwave.system import FixedEmbeddingComposite

import gonogo
import qpu_run

NS, RHOS, N_INST = [10, 18], [0.5, 1.0], 8
LOGLAMS = qpu_run.PHASES["main"]["loglams"]
EMB_SIZE = {10: 14, 18: 30}  # long chains: first n chains of a K_size embedding


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=float, default=80, help="max QPU access seconds for this invocation")
    ap.add_argument("--month-cap", type=float, default=560, help="max QPU seconds per calendar month (ledger)")
    ap.add_argument("--limit", type=int, default=None, help="run at most this many instances")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--solver", default="Advantage_system4")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = "dry" if args.dry_run else "qpu"
    out = args.out or f"{tag}_interleave.csv"
    ledger_path = f"{tag}_ledger.json" if args.dry_run else qpu_run.LEDGER
    month = time.strftime("%Y-%m")
    done = set(pd.read_csv(out, usecols=["inst_id"]).inst_id) if os.path.exists(out) else set()

    tasks = []
    for n in NS:
        for rho in RHOS:
            M = max(1, round(rho * n))
            for idx in range(N_INST):
                if f"n{n}_M{M}_{idx}" not in done:
                    tasks.append((n, M, idx, int(np.random.SeedSequence([args.seed, n, M, idx]).generate_state(1)[0])))
    child, graph, chip = qpu_run.make_sampler(args.dry_run, None if args.dry_run else args.solver)
    emb_cache = qpu_run.load_ledger(f"{tag}_embeddings.json")
    samplers = {}
    for n in NS:
        for emb, size in [("short", None), ("long", EMB_SIZE[n])]:
            e = qpu_run.clique_embedding(n, graph, emb_cache, size)
            samplers[n, emb] = FixedEmbeddingComposite(child, e)
            print(f"n={n} {emb}: chain length {np.mean([len(c) for c in e.values()]):.2f}")
    print(f"{chip}: {len(tasks)} instances to run, {len(done)} done; "
          f"month {month} used {qpu_run.load_ledger(ledger_path).get(month, 0) / 1e6:.1f}s", flush=True)

    used_us, ran = 0, 0
    per_inst = 2 * len(LOGLAMS) * 2 * 40_000  # first-instance estimate of QPU time (us)
    for n, M, idx, seed in tasks:
        if args.limit is not None and ran >= args.limit:
            break
        iid = f"n{n}_M{M}_{idx}"
        next_est = used_us / ran if ran else per_inst
        month_used = qpu_run.load_ledger(ledger_path).get(month, 0)
        if used_us + next_est > args.budget * 1e6 or month_used + next_est > args.month_cap * 1e6:
            print(f"stopping before {iid}: budget would be exceeded (this run {used_us / 1e6:.1f}s, "
                  f"month {month_used / 1e6:.1f}s)")
            break
        inst = gonogo.make_instance(n, M, seed)
        rng = np.random.default_rng(seed + 53)
        jobs = [(float(ll), emb) for ll in LOGLAMS for emb in ("short", "long")]
        rng.shuffle(jobs)
        w = 1 << np.arange(n)
        rows, inst_us = [], 0
        try:
            for loglam, emb in jobs:
                h, J = gonogo.qubo_to_ising(inst["U"] + 10.0 ** loglam * inst["P"])
                smp = samplers[n, emb]
                row = dict(inst_id=iid, n=n, M=M, cond="QPU", phase="interleave", emb=emb, loglam=loglam,
                           chain_len=float(np.mean([len(c) for c in smp.embedding.values()])),
                           lam_exact=inst["lam_exact"], lam_inst=inst["lam_inst"], lam_wc=inst["lam_wc"], chip=chip)
                cbs, job_us = [], 0
                for half in "ab":
                    S, us, cb = qpu_run.sample_gauged(smp, h, J, rng, args.dry_run)
                    x = ((S + 1) // 2) @ w
                    f = inst["feas"][x]
                    row[f"succ_{half}"] = int((f & (inst["E"][x] == inst["E_star"])).sum())
                    row[f"feas_{half}"] = int(f.sum())
                    row[f"reads_{half}"] = len(x)
                    cbs.append(cb)
                    job_us += us
                row.update(chain_break=float(np.mean(cbs)), qpu_us=job_us)
                rows.append(row)
                inst_us += job_us
        except Exception as e:  # e.g. out of solver access time: log the time spent, keep finished instances, stop
            qpu_run.add_to_ledger(ledger_path, month, sum(r["qpu_us"] for r in rows))
            print(f"stopped during {iid} ({type(e).__name__}: {e}); instance not saved")
            break
        pd.DataFrame(rows).to_csv(out, mode="a", index=False, header=not os.path.exists(out))
        total = qpu_run.add_to_ledger(ledger_path, month, inst_us)
        used_us += inst_us
        ran += 1
        print(f"{iid}: {inst_us / 1e6:.2f}s; this run {used_us / 1e6:.1f}s, month {total / 1e6:.1f}s", flush=True)

    if os.path.exists(out):
        df = pd.read_csv(out)
        df["success"] = (df.succ_a + df.succ_b) / (df.reads_a + df.reads_b)
        print("\nmean success by n and embedding")
        print(df.pivot_table(index="n", columns="emb", values="success").round(4).to_string())
        print("\nmean chain-break fraction")
        print(df.pivot_table(index="n", columns="emb", values="chain_break").round(4).to_string())


if __name__ == "__main__":
    main()
