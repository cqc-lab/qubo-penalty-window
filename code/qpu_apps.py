"""TSP and knapsack encodings of apps.py on the QPU, with the window-existence index estimated from samples.

For every instance, est (estimator.py) selects lambda on the QPU. From its 1000 selection reads alone we form
  g_hat   : gap between the two lowest distinct feasible objective values seen,
  lam_hat : est's estimate of lambda_exact,
  Gamma_hat = beta g_hat / s(1.5 lam_hat), with beta = 7 / (mean chain length), the hardware relation of the paper,
and compare it with the exact Gamma (enumeration, same beta). Each instance is then scored on the QPU at the
lambda of est, of the oracle 1.5 lambda_exact, of the conventional rule and of lambda_inst (1000 reads each), and
on a short curve in lambda / lambda_exact (500 reads per point). Clique embeddings, Ocean's default chain strength.

    python qpu_apps.py --dry-run --limit 1
    python qpu_apps.py --limit 1 --budget 3 --solver Advantage_system6
    python qpu_apps.py --budget 30 --solver Advantage_system6
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from dwave.system import FixedEmbeddingComposite

import apps
import estimator
import gonogo
import qpu_estimator
import qpu_run

TASKS = [("tsp", 5, i) for i in range(10)] + [("knapsack", 10, i) for i in range(10)]
CURVE = [-0.3, 0.0, 0.18, 0.5, 1.0, 1.5]  # log10(lambda / lambda_exact)
EMB_FILE = "qpu_apps_embeddings.json"


class Recording(qpu_estimator.QPUDraw):
    """QPUDraw that also keeps every (objective, violation) pair it returns."""

    def __init__(self, samplers, dry_run):
        super().__init__(samplers, dry_run)
        self.E, self.V = [], []

    def __call__(self, *a, **k):
        E, V, hit = super().__call__(*a, **k)
        self.E.append(E)
        self.V.append(V)
        return E, V, hit


def scale(inst, lam):
    h, J = gonogo.qubo_to_ising(inst["U"] + lam * inst["P"])
    return max(np.abs(h).max() / gonogo.H_RANGE, np.abs(J).max() / gonogo.J_RANGE)


def gap(E):
    u = np.unique(np.round(E, 9))
    return float(u[1] - u[0]) if len(u) > 1 else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=float, default=30, help="max QPU access seconds for this invocation")
    ap.add_argument("--month-cap", type=float, default=560, help="max QPU seconds per calendar month (ledger)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--solver", default="Advantage_system6")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = "dry" if args.dry_run else "qpu"
    out = args.out or f"{tag}_apps.csv"
    ledger_path = f"{tag}_ledger.json" if args.dry_run else qpu_run.LEDGER
    month = time.strftime("%Y-%m")
    done = set(pd.read_csv(out, usecols=["inst_id"]).inst_id) if os.path.exists(out) else set()
    child, graph, chip = qpu_run.make_sampler(args.dry_run, None if args.dry_run else args.solver)
    emb_path = EMB_FILE + (".dry" if args.dry_run else "")
    cache = qpu_run.load_ledger(emb_path)
    print(f"{chip}: {len(TASKS) - len(done)} instances to run; "
          f"month {month} used {qpu_run.load_ledger(ledger_path).get(month, 0) / 1e6:.1f}s", flush=True)

    used_us, ran = 0, 0
    for problem, size, idx in TASKS:
        inst, meta, seed = apps.build(problem, size, idx)
        if meta["inst_id"] in done:
            continue
        if args.limit is not None and ran >= args.limit:
            break
        next_est = used_us / ran if ran else 1.5e6
        month_used = qpu_run.load_ledger(ledger_path).get(month, 0)
        if used_us + next_est > args.budget * 1e6 or month_used + next_est > args.month_cap * 1e6:
            print(f"stopping: budget would be exceeded (this run {used_us / 1e6:.1f}s, month {month_used / 1e6:.1f}s)")
            break
        n = meta["n"]
        emb = qpu_run.clique_embedding(n, graph, cache)
        qpu_run.json.dump(cache, open(emb_path, "w"))
        ell = float(np.mean([len(c) for c in emb.values()]))
        draw = Recording({n: FixedEmbeddingComposite(child, emb)}, args.dry_run)
        estimator.draw = draw
        rng = np.random.default_rng(seed + 31)
        est, _ = estimator.estimate(inst, None, rng, None)
        E, V = np.concatenate(draw.E), np.concatenate(draw.V)
        select_us = draw.us
        beta = 7.0 / ell
        lam_hat = 10.0 ** est / 1.5
        g_hat = gap(E[V == 0]) if (V == 0).any() else np.nan
        g_true = gap(inst["E"][inst["feas"]])
        lex = inst["lam_exact"]
        gamma_hat = beta * g_hat / scale(inst, 1.5 * lam_hat)
        gamma = beta * g_true / scale(inst, 1.5 * lex) if lex > 0 else np.nan
        chosen = dict(est=est, oracle=np.log10(max(1.5 * lex, 1e-3)), conv=np.log10(inst["lam_wc"]),
                      inst=np.log10(inst["lam_inst"]))
        jobs = [("method", m, round(float(ll), 3)) for m, ll in chosen.items()]
        jobs += [("curve", c, round(float(np.log10(lex) + c), 3)) for c in CURVE] if lex > 0 else []
        rng.shuffle(jobs)
        scored, rows = {}, []
        for kind, key, ll in jobs:
            reads, progs = (estimator.READS_EVAL, 4) if kind == "method" else (500, 2)
            if (ll, reads) not in scored:
                _, _, hit = draw(inst, ll, None, reads, rng, None, n_prog=progs)
                scored[ll, reads] = hit.mean()
            rows.append(dict(meta, kind=kind, method=key if kind == "method" else "curve",
                             rel=round(ll - np.log10(lex), 2) if lex > 0 else np.nan, loglam=ll,
                             success=scored[ll, reads]))
        info = dict(chain_len=ell, beta=beta, lam_exact=lex, lam_hat=lam_hat, g=g_true, g_hat=g_hat, gamma=gamma,
                    gamma_hat=gamma_hat, chip=chip, select_us=select_us, qpu_us_instance=draw.us)
        rows = [dict(r, **info) for r in rows]
        pd.DataFrame(rows).to_csv(out, mode="a", index=False, header=not os.path.exists(out))
        total = qpu_run.add_to_ledger(ledger_path, month, draw.us)
        used_us += draw.us
        ran += 1
        print(f"{meta['inst_id']}: n={n} chain {ell:.1f}, Gamma {gamma:.3g} vs Gamma_hat {gamma_hat:.3g}, "
              f"{draw.us / 1e6:.2f}s; this run {used_us / 1e6:.1f}s, month {total / 1e6:.1f}s", flush=True)

    if os.path.exists(out):
        df = pd.read_csv(out)
        m = df[df.kind == "method"]
        print("\nmean success by problem and method")
        print(m.pivot_table(index="problem", columns="method", values="success").round(3).to_string())
        c = df[df.kind == "curve"]
        print("\nmean success on the curve, log10(lambda / lambda_exact)")
        print(c.pivot_table(index="rel", columns="problem", values="success").round(3).to_string())
        g = df.drop_duplicates("inst_id")
        print("\nGamma (exact) and Gamma_hat (from est's reads), log10 medians")
        print(g.groupby("problem")[["gamma", "gamma_hat"]].agg(lambda x: np.log10(x).median()).round(2).to_string())


if __name__ == "__main__":
    main()
