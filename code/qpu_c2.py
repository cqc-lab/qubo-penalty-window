"""Held-out check on the QPU: penalty selection on one-hot-constrained instances, margin frozen at 1.5.

The instances are extra.py's "onehot" family (same seeds, so they pair with the emulation results). Every method
of qpu_estimator.py and the Garcia et al. variants of qpu_garcia.py is run on each instance back to back;
methods that choose the same lambda share one evaluation. Finished instances are skipped on restart.

    python qpu_c2.py --dry-run
    python qpu_c2.py --budget 25
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from dwave.system import FixedEmbeddingComposite

import estimator
import extra
import qpu_estimator
import qpu_garcia
import qpu_run

NS, N_INST = [12, 15, 18], 10
METHODS = qpu_estimator.METHODS + ["g_bin", "g_bin15", "g_bin4", "g_bin4_15"]


def run_instance(inst, seed, draw):
    estimator.draw = draw
    rng = np.random.default_rng(seed + 11)
    est, _ = estimator.estimate(inst, None, rng, None)
    seq = estimator.sequential(inst, None, rng, None)
    chosen = dict(est=(est, 4), est_nomargin=(est - estimator.MARGIN, 4), seq=(seq, 4),
                  seq15=(seq + estimator.MARGIN, 4), grid=(estimator.grid_search(inst, None, rng, None), 4),
                  oracle=(np.log10(max(1.5 * inst["lam_exact"], 1e-3)), 0), inst=(np.log10(inst["lam_inst"]), 0))
    g_bin, g_bin4 = qpu_garcia.garcia_binary_prefix(inst, np.random.default_rng(seed + 23))
    chosen.update(g_bin=g_bin, g_bin15=(g_bin[0] + estimator.MARGIN, g_bin[1]), g_bin4=g_bin4,
                  g_bin4_15=(g_bin4[0] + estimator.MARGIN, g_bin4[1]))
    select_us = draw.us
    scored = {}
    for m in METHODS:
        ll = round(float(chosen[m][0]), 3)
        if ll not in scored:
            _, _, hit = draw(inst, ll, None, estimator.READS_EVAL, rng, None, n_prog=qpu_estimator.EVAL_PROGS)
            scored[ll] = hit.mean()
    return chosen, scored, select_us, draw.us - select_us


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=float, default=25, help="max QPU access seconds for this invocation")
    ap.add_argument("--month-cap", type=float, default=560, help="max QPU seconds per calendar month (ledger)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--solver", default="Advantage_system4")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = "dry" if args.dry_run else "qpu"
    out = args.out or f"{tag}_c2.csv"
    ledger_path = f"{tag}_ledger.json" if args.dry_run else qpu_run.LEDGER
    month = time.strftime("%Y-%m")
    done = set(pd.read_csv(out, usecols=["inst_id"]).inst_id) if os.path.exists(out) else set()
    # interleave sizes so a partial run covers all of them
    tasks = [(n, idx) for idx in range(N_INST) for n in NS]

    child, graph, chip = qpu_run.make_sampler(args.dry_run, None if args.dry_run else args.solver)
    emb_path = f"{tag}_embeddings.json"
    emb_cache = qpu_run.load_ledger(emb_path)
    samplers = {n: FixedEmbeddingComposite(child, qpu_run.clique_embedding(n, graph, emb_cache)) for n in NS}
    qpu_run.json.dump(emb_cache, open(emb_path, "w"))
    print(f"{chip}: {len(tasks) - len(done)} instances to run, {len(done)} done; chain lengths "
          f"{ {n: round(np.mean([len(c) for c in samplers[n].embedding.values()]), 2) for n in NS} }; "
          f"month {month} used {qpu_run.load_ledger(ledger_path).get(month, 0) / 1e6:.1f}s", flush=True)

    used_us, ran = 0, 0
    for n, idx in tasks:
        inst, seed, meta = extra.make("onehot", n, None, idx)
        if meta["inst_id"] in done:
            continue
        next_est = used_us / ran if ran else 3e6
        month_used = qpu_run.load_ledger(ledger_path).get(month, 0)
        if used_us + next_est > args.budget * 1e6 or month_used + next_est > args.month_cap * 1e6:
            print(f"stopping before {meta['inst_id']}: budget would be exceeded "
                  f"(this run {used_us / 1e6:.1f}s, month {month_used / 1e6:.1f}s)")
            break
        draw = qpu_estimator.QPUDraw(samplers, args.dry_run)
        try:
            chosen, scored, sel_us, eval_us = run_instance(inst, seed, draw)
        except Exception as e:  # e.g. out of solver access time
            qpu_run.add_to_ledger(ledger_path, month, draw.us)
            print(f"stopped during {meta['inst_id']} ({type(e).__name__}: {e}); instance not saved")
            break
        total = qpu_run.add_to_ledger(ledger_path, month, draw.us)
        used_us += draw.us
        ran += 1
        lex = inst["lam_exact"]
        rows = [dict(meta, cond="QPU", chip=chip, method=m, loglam=chosen[m][0], rounds=chosen[m][1],
                     success=scored[round(float(chosen[m][0]), 3)],
                     log_ratio_exact=chosen[m][0] - np.log10(lex) if lex > 0 else np.nan,
                     select_us=sel_us, eval_us=eval_us) for m in METHODS]
        pd.DataFrame(rows).to_csv(out, mode="a", index=False, header=not os.path.exists(out))
        print(f"{meta['inst_id']}: {draw.us / 1e6:.2f}s (select {sel_us / 1e6:.2f}, eval {eval_us / 1e6:.2f}, "
              f"{len(scored)} distinct lambdas); this run {used_us / 1e6:.1f}s, month {total / 1e6:.1f}s", flush=True)

    if os.path.exists(out):
        df = pd.read_csv(out)
        print("\nmean success")
        print(df.pivot_table(index="n", columns="method", values="success")[METHODS].round(3).to_string())


if __name__ == "__main__":
    main()
