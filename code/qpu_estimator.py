"""Lambda-selection methods of estimator.py run on a D-Wave QPU, with a hard budget on QPU access time.

Instances are the qpu_run.py main-phase instances (same seeds), so each chosen lambda can also be read
against the measured QPU success curve. The selection logic is estimator.py's, with its sampler swapped
for the QPU; methods that choose the same lambda share one evaluation (QPU draws cannot be replayed).

    python qpu_estimator.py --dry-run                  # SA on a Pegasus-shaped graph, no QPU time used
    python qpu_estimator.py --budget 150               # real QPU
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from dwave.system import FixedEmbeddingComposite

import estimator
import gonogo
import qpu_run

NS, RHOS, N_INST = [10, 14, 18], [0.5, 1.0], 8
EVAL_PROGS = 4  # READS_EVAL split over this many gauged programmings
METHODS = ["est", "est_nomargin", "seq", "seq15", "grid", "oracle", "inst"]


class QPUDraw:
    """Drop-in for estimator.draw that samples the QPU and meters access time."""

    def __init__(self, samplers, dry_run):
        self.samplers, self.dry_run, self.us = samplers, dry_run, 0

    def __call__(self, inst, loglam, cond, reads, rng, sampler, n_prog=1):
        n = inst["U"].shape[0]
        h, J = gonogo.qubo_to_ising(inst["U"] + 10.0 ** loglam * inst["P"])
        parts = []
        for _ in range(n_prog):
            S, us, _ = qpu_run.sample_gauged(self.samplers[n], h, J, rng, self.dry_run, reads // n_prog)
            parts.append(S)
            self.us += us
        S = np.concatenate(parts)
        k = ((S + 1) // 2) @ (1 << np.arange(n))
        return inst["E"][k], inst["V"][k], inst["feas"][k] & (inst["E"][k] == inst["E_star"])


def run_instance(inst, seed, draw):
    estimator.draw = draw  # estimate / sequential / grid_search look up the module-level draw
    rng = np.random.default_rng(seed + 11)
    est, _ = estimator.estimate(inst, None, rng, None)
    seq = estimator.sequential(inst, None, rng, None)
    chosen = dict(est=est, est_nomargin=est - estimator.MARGIN, seq=seq, seq15=seq + estimator.MARGIN,
                  grid=estimator.grid_search(inst, None, rng, None),
                  oracle=np.log10(max(1.5 * inst["lam_exact"], 1e-3)), inst=np.log10(inst["lam_inst"]))
    select_us = draw.us
    scored = {}
    for method in METHODS:
        ll = round(float(chosen[method]), 3)
        if ll not in scored:
            _, _, hit = draw(inst, ll, None, estimator.READS_EVAL, rng, None, n_prog=EVAL_PROGS)
            scored[ll] = hit.mean()
    return chosen, scored, select_us, draw.us - select_us


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=float, default=150, help="max QPU access seconds for this invocation")
    ap.add_argument("--month-cap", type=float, default=560, help="max QPU seconds per calendar month (ledger)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--solver", default=None)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = "dry" if args.dry_run else "qpu"
    out = args.out or f"{tag}_estimator.csv"
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

    child, graph, chip = qpu_run.make_sampler(args.dry_run, args.solver)
    emb_path = f"{tag}_embeddings.json"
    emb_cache = qpu_run.load_ledger(emb_path)
    samplers = {n: FixedEmbeddingComposite(child, qpu_run.clique_embedding(n, graph, emb_cache)) for n in NS}
    print(f"{chip}: {len(tasks)} instances to run, {len(done)} already done; "
          f"month {month} used {qpu_run.load_ledger(ledger_path).get(month, 0) / 1e6:.1f}s", flush=True)

    used_us = 0
    for k, (n, M, idx, seed) in enumerate(tasks, 1):
        next_est = used_us / (k - 1) if k > 1 else 4e6
        month_used = qpu_run.load_ledger(ledger_path).get(month, 0)
        if used_us + next_est > args.budget * 1e6 or month_used + next_est > args.month_cap * 1e6:
            print(f"stopping before instance {k}: budget would be exceeded "
                  f"(this run {used_us / 1e6:.1f}s, month {month_used / 1e6:.1f}s)")
            break
        inst = gonogo.make_instance(n, M, seed)
        inst["V"] = estimator.violations(n, inst)
        draw = QPUDraw(samplers, args.dry_run)
        chosen, scored, sel_us, eval_us = run_instance(inst, seed, draw)
        total = qpu_run.add_to_ledger(ledger_path, month, draw.us)
        used_us += draw.us
        lam_exact = inst["lam_exact"]
        rows = [dict(inst_id=f"n{n}_M{M}_{idx}", n=n, M=M, cond="QPU", method=m, loglam=chosen[m],
                     success=scored[round(float(chosen[m]), 3)],
                     log_ratio_exact=chosen[m] - np.log10(lam_exact) if lam_exact > 0 else np.nan,
                     select_us=sel_us, eval_us=eval_us) for m in METHODS]
        pd.DataFrame(rows).to_csv(out, mode="a", index=False, header=not os.path.exists(out))
        print(f"[{k}/{len(tasks)}] n{n}_M{M}_{idx}: {draw.us / 1e6:.2f}s "
              f"(select {sel_us / 1e6:.2f}, eval {eval_us / 1e6:.2f}, {len(scored)} distinct lambdas); "
              f"this run {used_us / 1e6:.1f}s, month {total / 1e6:.1f}s", flush=True)

    if os.path.exists(out):
        df = pd.read_csv(out)
        print("\nmean success")
        print(df.pivot_table(index=["n", "M"], columns="method", values="success", aggfunc="mean")
              [METHODS].round(3).to_string())
        print("\nmedian log10(chosen / lambda_exact)")
        print(df.pivot_table(index=["n", "M"], columns="method", values="log_ratio_exact", aggfunc="median")
              [METHODS].round(3).to_string())


if __name__ == "__main__":
    main()
