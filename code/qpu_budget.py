"""Equal-budget comparison on the large sparse instances: est with its selection reads against a fixed weight.

For each of the 60 sparse instances of qpu_large.py (same minorminer embeddings, Ocean's default chain strength):
  est   : the 4 x 250 selection reads of large.estimate, then 2 x 250 reads at the chosen weight (1500 reads in all);
  inst  : 6 x 250 reads at the coefficient-based weight lambda_inst (1500 reads);
  ref   : 2 x 250 reads at 1.5 lam_ref, for reference.
  seq15 : (optional, --methods) the 4 x 250 selection reads of large.sequential, then 2 x 250 reads at its choice
          times 1.5 (1500 reads in all).
Every read is kept, including the selection reads, so for each method we know whether a solution within 1% of
E_ref (and within 5%) appeared within the same total number of reads. The scoring jobs of all methods are
interleaved in random order per instance; the order of the two selection procedures is random as well.

Protocol of the Advantage2 run, fixed before it was run (2026-11-17): methods est, seq15 and inst; margin 1.5;
1500 reads per method including selection; criterion: a feasible read within 1% of E_ref (large.csv) among all
reads of the method; embeddings from embed_large.py on the Advantage2 working graph.

    python qpu_budget.py --dry-run --limit 1
    python qpu_budget.py --budget 60 --solver Advantage_system6
    python qpu_budget.py --budget 110 --solver Advantage2_system1 --methods est,seq15,inst --out qpu_budget_adv2.csv
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from dwave.system import FixedEmbeddingComposite

import large
import qpu_large
import qpu_run

READS = 250


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="large.csv")
    ap.add_argument("--budget", type=float, default=60)
    ap.add_argument("--month-cap", type=float, default=560)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--solver", default="Advantage_system6")
    ap.add_argument("--out", default=None)
    ap.add_argument("--methods", default="est,inst,ref", help="comma-separated: est, seq15, inst, ref")
    args = ap.parse_args()
    qpu_large.DRY[0] = args.dry_run
    tag = "dry" if args.dry_run else "qpu"
    out = args.out or f"{tag}_budget.csv"
    ledger_path = f"{tag}_ledger.json" if args.dry_run else qpu_run.LEDGER
    month = time.strftime("%Y-%m")
    refs = pd.read_csv(args.ref).drop_duplicates(["family", "n", "M", "idx"])
    refs = refs[refs.family == "sparse"].sort_values(["n", "M", "idx"]).to_dict("records")
    done = set(pd.read_csv(out).inst_id) if os.path.exists(out) else set()
    child, graph, chip = qpu_run.make_sampler(args.dry_run, None if args.dry_run else args.solver)
    emb_file = qpu_large.EMB_FILE + (".dry" if args.dry_run else "")
    cache = qpu_run.load_ledger(emb_file)
    print(f"{chip}: month {month} used {qpu_run.load_ledger(ledger_path).get(month, 0) / 1e6:.1f}s", flush=True)

    used_us, ran = 0, 0
    for ref in refs:
        n, M, idx = int(ref["n"]), int(ref["M"]), int(ref["idx"])
        iid = f"sparse_{n}_{M}_{idx}"
        if iid in done:
            continue
        if args.limit is not None and ran >= args.limit:
            break
        next_est = used_us / ran if ran else 14 * 60_000
        month_used = qpu_run.load_ledger(ledger_path).get(month, 0)
        if used_us + next_est > args.budget * 1e6 or month_used + next_est > args.month_cap * 1e6:
            print(f"stopping before {iid}: budget would be exceeded (this run {used_us / 1e6:.1f}s)")
            break
        inst, seed = large.make_large("sparse", n, M, idx)
        key = f"{chip}:sparse_{n}_{M}_{idx}"
        emb = qpu_large.embedding_for(inst, key, "sparse", n, graph, cache, seed)
        sampler = FixedEmbeddingComposite(child, emb)
        rng = np.random.default_rng(int(np.random.SeedSequence([2026, 91, n, M, idx]).generate_state(1)[0]))
        qpu_large.USED_US[0] = 0
        qpu_large.CB.clear()
        one = dict(n_prog=1, reads=READS)
        methods = args.methods.split(",")
        chosen = {"inst": np.log10(inst["lam_inst"]), "ref": np.log10(1.5 * ref["lam_ref"])}
        if methods == ["est", "inst", "ref"]:
            # the original run: est selection, then its scoring reads interleaved with the fixed weights
            pool = []
            chosen["est"] = large.estimate(inst, rng, sampler, pool)
            draws = {"est": list(pool), "inst": [], "ref": []}
        else:
            draws = {m: [] for m in methods}
            sel = [m for m in ("est", "seq15") if m in methods]
            for m in [sel[i] for i in rng.permutation(len(sel))]:
                pool = []
                if m == "est":
                    chosen[m] = large.estimate(inst, rng, sampler, pool)
                else:
                    chosen[m] = large.sequential(inst, rng, sampler, pool) + np.log10(1.5)
                draws[m] = list(pool)
        jobs = []
        for m in methods:
            jobs += [(m, chosen[m])] * (6 if m == "inst" else 2)
        rng.shuffle(jobs)
        for meth, ll in jobs:
            draws[meth].append(qpu_large.qpu_draw(inst, 10.0 ** ll, None, one, rng, sampler))
        E_new = min((E[V == 0].min() for d in draws.values() for E, V in d if (V == 0).any()), default=np.inf)
        E_ref = min(ref["E_ref"], E_new)
        tol = abs(E_ref)
        rows = []
        for meth, ds in draws.items():
            E = np.concatenate([d[0] for d in ds])
            V = np.concatenate([d[1] for d in ds])
            ok = V == 0
            near1 = ok & (E <= E_ref + 0.01 * tol)
            near5 = ok & (E <= E_ref + 0.05 * tol)
            rows.append(dict(inst_id=iid, n=n, M=M, idx=idx, method=meth, reads=len(E),
                             loglam=chosen[meth],
                             any1=bool(near1.any()), n1=int(near1.sum()), any5=bool(near5.any()), n5=int(near5.sum()),
                             feas=float(ok.mean()), E_ref=E_ref, E_ref_improved=bool(E_new < ref["E_ref"]),
                             lam_ref=ref["lam_ref"], chip=chip, chain_break=float(np.mean(qpu_large.CB)),
                             qpu_us_instance=qpu_large.USED_US[0]))
        pd.DataFrame(rows).to_csv(out, mode="a", index=False, header=not os.path.exists(out))
        total = qpu_run.add_to_ledger(ledger_path, month, qpu_large.USED_US[0])
        used_us += qpu_large.USED_US[0]
        ran += 1
        print(f"{iid}: {qpu_large.USED_US[0] / 1e6:.2f}s; this run {used_us / 1e6:.1f}s, month {total / 1e6:.1f}s",
              flush=True)

    if os.path.exists(out):
        df = pd.read_csv(out)
        print("\nfraction of instances with a solution within 1% / 5% of E_ref, and mean count, by method")
        print(df.groupby("method")[["reads", "any1", "n1", "any5", "n5", "feas"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
