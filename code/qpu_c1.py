"""QPU sweep in lambda / lambda_exact under two chain-strength settings, paired within each instance.

  default : Ocean's default chain strength (uniform torque compensation), as in qpu_run.py
  scaled  : chain strength = s(lambda), the logical auto-scale factor. Chain couplers then program at -1 and the
            embedded problem is scaled exactly like the logical one, so s(lambda) is the hardware's scale.

Instances are the qpu_run.py main-phase instances (same seeds) with lambda_exact > 0. Every (lambda, setting)
point of an instance is run before the next instance starts, in shuffled order, so a run stopped early still
leaves complete pairs. Finished instances are skipped on restart.

    python qpu_c1.py --ns 10 --dry-run
    python qpu_c1.py --ns 10 --limit 1 --budget 5      # stage 0: one instance
    python qpu_c1.py --ns 10 --budget 60
"""
import argparse
import os
import time

import dimod
import numpy as np
import pandas as pd
from dwave.system import FixedEmbeddingComposite

import gonogo
import qpu_run

RHOS, N_INST = [0.5, 1.0], 8
REL = [-0.3, -0.1, 0.0, 0.1, 0.18, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]  # log10(lambda / lambda_exact)
SETTINGS = ["default", "scaled"]


def sample(sampler, h, J, rng, dry_run, chain_strength, reads=qpu_run.READS_PER_HALF):
    """One gauged programming, as qpu_run.sample_gauged, with an explicit chain strength (None = default)."""
    n = len(h)
    g = rng.choice([-1, 1], n)
    hg, Jg = h * g, J * np.outer(g, g)
    bqm = dimod.BinaryQuadraticModel(hg, Jg, 0.0, "SPIN")
    if dry_run:
        ss = sampler.sample(bqm, num_reads=reads, num_sweeps=1000, chain_strength=chain_strength,
                            beta_range=(0.1, 5.0 / max(np.abs(hg).max() / 4, np.abs(Jg).max())),
                            seed=int(rng.integers(2**31 - 1)))
        us = qpu_run.DRY_RUN_US_PER_JOB
    else:
        ss = sampler.sample(bqm, num_reads=reads, answer_mode="raw", chain_strength=chain_strength,
                            label="penalty-window-c1")
        us = int(ss.info["timing"]["qpu_access_time"])
    S = ss.record.sample[:, np.argsort(np.asarray(list(ss.variables)))] * g
    return S, us, float(np.mean(ss.record.chain_break_fraction))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ns", type=int, nargs="+", default=[10, 18])
    ap.add_argument("--budget", type=float, default=60, help="max QPU access seconds for this invocation")
    ap.add_argument("--month-cap", type=float, default=560, help="max QPU seconds per calendar month (ledger)")
    ap.add_argument("--limit", type=int, default=None, help="run at most this many instances")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--solver", default="Advantage_system4")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = "dry" if args.dry_run else "qpu"
    out = args.out or f"{tag}_c1.csv"
    ledger_path = f"{tag}_ledger.json" if args.dry_run else qpu_run.LEDGER
    month = time.strftime("%Y-%m")
    done = set(pd.read_csv(out, usecols=["inst_id"]).inst_id) if os.path.exists(out) else set()

    tasks = []
    for n in args.ns:
        for rho in RHOS:
            M = max(1, round(rho * n))
            for idx in range(N_INST):
                if f"n{n}_M{M}_{idx}" not in done:
                    tasks.append((n, M, idx, int(np.random.SeedSequence([args.seed, n, M, idx]).generate_state(1)[0])))
    child, graph, chip = qpu_run.make_sampler(args.dry_run, None if args.dry_run else args.solver)
    emb_cache = qpu_run.load_ledger(f"{tag}_embeddings.json")
    samplers = {n: FixedEmbeddingComposite(child, qpu_run.clique_embedding(n, graph, emb_cache)) for n in args.ns}
    print(f"{chip}: {len(tasks)} instances to check, {len(done)} done; "
          f"month {month} used {qpu_run.load_ledger(ledger_path).get(month, 0) / 1e6:.1f}s", flush=True)

    used_us, ran = 0, 0
    per_inst = 2 * len(REL) * len(SETTINGS) * 120_000  # first-instance estimate of QPU time (us)
    for n, M, idx, seed in tasks:
        iid = f"n{n}_M{M}_{idx}"
        inst = gonogo.make_instance(n, M, seed)
        if inst["lam_exact"] <= 0:
            continue
        if args.limit is not None and ran >= args.limit:
            break
        next_est = used_us / ran if ran else per_inst
        month_used = qpu_run.load_ledger(ledger_path).get(month, 0)
        if used_us + next_est > args.budget * 1e6 or month_used + next_est > args.month_cap * 1e6:
            print(f"stopping before {iid}: budget would be exceeded (this run {used_us / 1e6:.1f}s, "
                  f"month {month_used / 1e6:.1f}s)")
            break
        rng = np.random.default_rng(seed + 41)
        jobs = [(r, s) for r in REL for s in SETTINGS]
        rng.shuffle(jobs)
        w = 1 << np.arange(n)
        rows, inst_us = [], 0
        try:
            inst_us = run_jobs(jobs, inst, iid, n, M, chip, samplers, rng, args.dry_run, w, rows)
        except Exception as e:  # e.g. out of solver access time: keep what is done, log the time, stop
            qpu_run.add_to_ledger(ledger_path, month, sum(r["qpu_us"] for r in rows))
            print(f"stopped during {iid} ({type(e).__name__}: {e}); instance not saved")
            break
        # write the instance only when complete, so every saved instance has all pairs
        pd.DataFrame(rows).to_csv(out, mode="a", index=False, header=not os.path.exists(out))
        total = qpu_run.add_to_ledger(ledger_path, month, inst_us)
        used_us += inst_us
        ran += 1
        print(f"{iid}: {inst_us / 1e6:.2f}s; this run {used_us / 1e6:.1f}s, month {total / 1e6:.1f}s", flush=True)

    if os.path.exists(out):
        df = pd.read_csv(out)
        df["success"] = (df.succ_a + df.succ_b) / (df.reads_a + df.reads_b)
        print("\nmean success by log10(lambda / lambda_exact)")
        print(df.pivot_table(index="rel", columns=["n", "setting"], values="success").round(3).to_string())
        print("\nmean chain-break fraction")
        print(df.pivot_table(index="rel", columns=["n", "setting"], values="chain_break").round(4).to_string())


def run_jobs(jobs, inst, iid, n, M, chip, samplers, rng, dry_run, w, rows):
    inst_us = 0
    for rel, setting in jobs:
        loglam = float(np.log10(inst["lam_exact"]) + rel)
        h, J = gonogo.qubo_to_ising(inst["U"] + 10.0 ** loglam * inst["P"])
        s = max(np.abs(h).max() / gonogo.H_RANGE, np.abs(J).max() / gonogo.J_RANGE)
        cs = s if setting == "scaled" else None
        row = dict(inst_id=iid, n=n, M=M, cond="QPU", setting=setting, rel=rel, loglam=loglam,
                   lam_exact=inst["lam_exact"], scale=s, chip=chip)
        cbs, job_us = [], 0
        for half in "ab":
            S, us, cb = sample(samplers[n], h, J, rng, dry_run, cs)
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
    return inst_us


if __name__ == "__main__":
    main()
