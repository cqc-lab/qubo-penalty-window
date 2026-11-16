"""Garcia et al. (2022) sequential penalty methods on the D-Wave QPU, on the qpu_estimator.py instances.

g_std  : Algorithm 1 (w = 1, x10 until feasible), at most 10 rounds
g_bin  : Algorithm 3 (bisect [1, w_U] at round(sqrt(a b))), at most 10 rounds; g_bin4 = its first 4 rounds
*15    : the same choice times our 1.5 margin
est_re : the lambda est chose in qpu_estimator.csv, re-scored in this session as a drift control

One round = one programming of 250 reads, judged by its lowest penalized-energy sample. Methods that choose
the same lambda share one evaluation, as in qpu_estimator.py.

    python qpu_garcia.py --dry-run
    python qpu_garcia.py --budget 60
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from dwave.system import FixedEmbeddingComposite

import estimator
import gonogo
import qpu_estimator
import qpu_run

METHODS = ["est_re", "g_std", "g_bin", "g_bin15", "g_bin4", "g_bin4_15"]


def garcia_binary_prefix(inst, rng, iters=10, prefix=estimator.ROUNDS):
    """Algorithm 3 as in estimator.garcia_binary, also returning its result after the first `prefix` rounds."""
    a, b = 1.0, max(round(estimator.garcia_upper_bound(inst)), 1.0)
    best, rounds, at_prefix = None, 0, None
    for _ in range(iters):
        w = float(round(np.sqrt(a * b)))
        if rounds and w in (a, b):
            break
        rounds += 1
        feasible, f = estimator._garcia_run(inst, w, None, rng, None)
        if feasible:
            if best is None or (f, w) < best:
                best = (f, w)
            b = w
        else:
            a = w
        if rounds == prefix:
            at_prefix = (np.log10(best[1] if best else b), rounds)
    final = (np.log10(best[1] if best else b), rounds)
    return final, at_prefix or final


def run_instance(inst, seed, draw, est_ll):
    estimator.draw = draw  # _garcia_run looks up the module-level draw
    rng = np.random.default_rng(seed + 23)
    g_std = estimator.garcia_standard(inst, None, rng, None)
    g_bin, g_bin4 = garcia_binary_prefix(inst, rng)
    chosen = dict(est_re=(est_ll, estimator.ROUNDS), g_std=g_std, g_bin=g_bin,
                  g_bin15=(g_bin[0] + estimator.MARGIN, g_bin[1]), g_bin4=g_bin4,
                  g_bin4_15=(g_bin4[0] + estimator.MARGIN, g_bin4[1]))
    select_us = draw.us
    scored = {}
    for method in METHODS:
        ll = round(float(chosen[method][0]), 3)
        if ll not in scored:
            _, _, hit = draw(inst, ll, None, estimator.READS_EVAL, rng, None, n_prog=qpu_estimator.EVAL_PROGS)
            scored[ll] = hit.mean()
    return chosen, scored, select_us, draw.us - select_us


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=float, default=60, help="max QPU access seconds for this invocation")
    ap.add_argument("--month-cap", type=float, default=560, help="max QPU seconds per calendar month (ledger)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--solver", default=None)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--est", default="qpu_estimator.csv", help="earlier QPU run supplying est's lambda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = "dry" if args.dry_run else "qpu"
    out = args.out or f"{tag}_garcia.csv"
    ledger_path = f"{tag}_ledger.json" if args.dry_run else qpu_run.LEDGER
    month = time.strftime("%Y-%m")
    done = set(pd.read_csv(out, usecols=["inst_id"]).inst_id) if os.path.exists(out) else set()
    prev = pd.read_csv(args.est)
    est_ll = prev[prev.method == "est"].set_index("inst_id").loglam

    tasks = []
    for n in qpu_estimator.NS:
        for rho in qpu_estimator.RHOS:
            M = max(1, round(rho * n))
            for idx in range(qpu_estimator.N_INST):
                iid = f"n{n}_M{M}_{idx}"
                if iid not in done and iid in est_ll.index:
                    tasks.append((n, M, idx, int(np.random.SeedSequence([args.seed, n, M, idx]).generate_state(1)[0])))

    child, graph, chip = qpu_run.make_sampler(args.dry_run, args.solver)
    emb_cache = qpu_run.load_ledger(f"{tag}_embeddings.json")
    samplers = {n: FixedEmbeddingComposite(child, qpu_run.clique_embedding(n, graph, emb_cache))
                for n in qpu_estimator.NS}
    print(f"{chip}: {len(tasks)} instances to run, {len(done)} already done; "
          f"month {month} used {qpu_run.load_ledger(ledger_path).get(month, 0) / 1e6:.1f}s", flush=True)

    used_us = 0
    for k, (n, M, idx, seed) in enumerate(tasks, 1):
        next_est = used_us / (k - 1) if k > 1 else 3e6
        month_used = qpu_run.load_ledger(ledger_path).get(month, 0)
        if used_us + next_est > args.budget * 1e6 or month_used + next_est > args.month_cap * 1e6:
            print(f"stopping before instance {k}: budget would be exceeded "
                  f"(this run {used_us / 1e6:.1f}s, month {month_used / 1e6:.1f}s)")
            break
        iid = f"n{n}_M{M}_{idx}"
        inst = gonogo.make_instance(n, M, seed)
        inst["V"] = estimator.violations(n, inst)
        draw = qpu_estimator.QPUDraw(samplers, args.dry_run)
        chosen, scored, sel_us, eval_us = run_instance(inst, seed, draw, float(est_ll[iid]))
        total = qpu_run.add_to_ledger(ledger_path, month, draw.us)
        used_us += draw.us
        lam_exact = inst["lam_exact"]
        rows = [dict(inst_id=iid, n=n, M=M, cond="QPU", method=m, loglam=chosen[m][0], rounds=chosen[m][1],
                     success=scored[round(float(chosen[m][0]), 3)],
                     log_ratio_exact=chosen[m][0] - np.log10(lam_exact) if lam_exact > 0 else np.nan,
                     select_us=sel_us, eval_us=eval_us) for m in METHODS]
        pd.DataFrame(rows).to_csv(out, mode="a", index=False, header=not os.path.exists(out))
        print(f"[{k}/{len(tasks)}] {iid}: {draw.us / 1e6:.2f}s "
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
