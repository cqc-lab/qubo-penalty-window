"""Hardware scale of every QPU programming, with the spin-reversal gauge that was actually applied.

qpu_run.py and qpu_c1.py multiply h and J by a random gauge g (h_i g_i, J_ij g_i g_j) before every programming.
With the extended coupling range [-2, 1] and the signed per-qubit coupling limit, the QPU's auto-scale factor
depends on the gauge, so it has to be computed per programming, from the gauged problem.

The gauges are reconstructed from the scripts' random streams:
  qpu_run.py : rng = default_rng(seed + 7), shuffled job list, two gauges per job, in job order; a resumed
               invocation rebuilds the list of jobs not yet done and restarts the stream.
  qpu_c1.py  : per instance rng = default_rng(instance seed + 41), shuffled (rel, setting) jobs, two gauges per job.
Every reconstructed job sequence is checked against the row order of the CSV; the script stops on a mismatch.

For each programming (the two halves a, b of every row) it records the auto-scale factor, the term that sets it,
the strongest problem coupler and the chain couplers in hardware units, and the success of that half. The scaling
fit is then repeated with one point per programming.

    python embedded_gauge.py --workers 30
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import dimod
from dwave.embedding import embed_bqm
from dwave.embedding.chain_strength import uniform_torque_compensation

import gonogo
from embedded_scale import best_beta, master_curve
from embedded_autoscale import GRAPH, autoscale, load_adj

SEED = 2026
PHASES = {"main": ("qpu_main.csv", dict(ns=[10, 14, 18], rhos=[0.5, 1.0], n_inst=8), {}),
          "chain": ("qpu_chain.csv", dict(ns=[10, 18], rhos=[0.5, 1.0], n_inst=8), {10: "10of14", 18: "18of30"}),
          "bign": ("qpu_bign.csv", dict(ns=[20, 22], rhos=[0.5, 1.0], n_inst=8), {})}
LOGLAMS = np.round(np.linspace(0.5, 2.5, 15), 3)
C1_REL = [-0.3, -0.1, 0.0, 0.1, 0.18, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
C1_SETTINGS = ["default", "scaled"]


def inst_seed(n, M, idx):
    return int(np.random.SeedSequence([SEED, n, M, idx]).generate_state(1)[0])


def parse(iid):
    n, M, idx = iid.split("_")
    return int(n[1:]), int(M[1:]), int(idx)


def gauges_qpu_run(path, cfg):
    """List of (row index, half, gauge) for a qpu_run.py CSV, reconstructed invocation by invocation."""
    q = pd.read_csv(path)
    rows = list(zip(q.inst_id, q.loglam.round(3)))
    out, done, pos, invocations = [], set(), 0, 0
    while pos < len(rows):
        jobs = []
        for n in cfg["ns"]:
            for rho in cfg["rhos"]:
                M = max(1, round(rho * n))
                for idx in range(cfg["n_inst"]):
                    for ll in LOGLAMS:
                        if (f"n{n}_M{M}_{idx}", float(ll)) not in done:
                            jobs.append((f"n{n}_M{M}_{idx}", float(ll), n))
        rng = np.random.default_rng(SEED + 7)
        rng.shuffle(jobs)
        invocations += 1
        start = pos
        for iid, ll, n in jobs:
            if pos >= len(rows) or rows[pos] != (iid, ll):
                break
            for half in "ab":
                out.append((pos, half, rng.choice([-1, 1], n)))
            done.add((iid, ll))
            pos += 1
        if pos == start:
            raise RuntimeError(f"{path}: row {pos} {rows[pos]} does not match any reconstructed job order")
    return q, out, invocations


def gauges_c1(path):
    q = pd.read_csv(path)
    out = []
    for iid, g in q.groupby("inst_id", sort=False):
        n, M, idx = parse(iid)
        rng = np.random.default_rng(inst_seed(n, M, idx) + 41)
        jobs = [(r, s) for r in C1_REL for s in C1_SETTINGS]
        rng.shuffle(jobs)
        got = list(zip(g.rel.round(2), g.setting))
        if [(round(r, 2), s) for r, s in jobs] != got:
            raise RuntimeError(f"qpu_c1.csv: job order of {iid} does not match")
        for pos, _ in zip(g.index, jobs):
            for half in "ab":
                out.append((pos, half, rng.choice([-1, 1], n)))
    return q, out


def features(task):
    run, iid, emb_key, items = task  # items: (row, half, loglam, setting, gauge, succ, reads)
    n, M, idx = parse(iid)
    inst = gonogo.make_instance(n, M, inst_seed(n, M, idx))
    fe = np.unique(inst["E"][inst["feas"]])
    gap = float(fe[1] - fe[0]) if len(fe) > 1 else np.nan
    adj = load_adj(GRAPH)
    emb = {int(v): c for v, c in json.load(open("qpu_embeddings.json"))[emb_key].items()}
    chains = [set(c) for c in emb.values()]
    res = []
    for row, half, ll, setting, g, succ, reads in items:
        h, J = gonogo.qubo_to_ising(inst["U"] + 10.0 ** ll * inst["P"])
        s_log = max(np.abs(h).max() / gonogo.H_RANGE, np.abs(J).max() / gonogo.J_RANGE)
        hg, Jg = h * g, J * np.outer(g, g)
        bqm = dimod.BinaryQuadraticModel({i: hg[i] for i in range(n)},
                                         {(i, j): Jg[i, j] for i in range(n) for j in range(i + 1, n) if Jg[i, j]},
                                         0.0, "SPIN")
        cs = s_log if setting == "scaled" else uniform_torque_compensation(bqm, emb)
        tb = embed_bqm(bqm, emb, adj, chain_strength=cs)
        s_auto, term = autoscale(tb)
        jprob = max(abs(b) for (u, v), b in tb.quadratic.items() if not any(u in c and v in c for c in chains))
        res.append(dict(run=run, inst_id=iid, n=n, row=row, half=half, loglam=ll, setting=setting,
                        lam_exact=inst["lam_exact"], gap=gap, chain_len=np.mean([len(c) for c in emb.values()]),
                        s_log=s_log, s_auto=s_auto, auto_term=term, jprob_hw=jprob / s_auto, chain_hw=cs / s_auto,
                        success=succ / reads))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--out", default="embedded_gauge.csv")
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()

    if not os.path.exists(args.out):
        tasks = {}
        for run, (path, cfg, keys) in PHASES.items():
            q, gs, inv = gauges_qpu_run(path, cfg)
            print(f"{run}: {len(q)} rows, {len(gs)} programmings reconstructed in {inv} invocation(s)", flush=True)
            for pos, half, g in gs:
                r = q.loc[pos]
                n = int(r.n)
                tasks.setdefault((run, r.inst_id, keys.get(n, str(n))), []).append(
                    (pos, half, round(float(r.loglam), 3), "default", g, int(r[f"succ_{half}"]), int(r[f"reads_{half}"])))
        q, gs = gauges_c1("qpu_c1.csv")
        print(f"c1: {len(q)} rows, {len(gs)} programmings reconstructed", flush=True)
        for pos, half, g in gs:
            r = q.loc[pos]
            tasks.setdefault(("c1", r.inst_id, str(int(r.n))), []).append(
                (pos, half, float(r.loglam), r.setting, g, int(r[f"succ_{half}"]), int(r[f"reads_{half}"])))
        with Pool(args.workers) as pool:
            res = pool.map(features, [(k[0], k[1], k[2], v) for k, v in tasks.items()])
        pd.DataFrame([x for rs in res for x in rs]).to_csv(args.out, index=False)

    f = pd.read_csv(args.out)
    rel = f.loglam - np.log10(f.lam_exact.where(f.lam_exact > 0))
    f["region"] = np.where(rel >= np.log10(2), "ge2", np.where(rel.abs() <= 0.25, "near", "other"))
    f["ratio"] = f.s_auto / f.s_log
    pd.set_option("display.width", 200)
    print("term that sets the hardware scale (fraction of programmings):")
    print(pd.crosstab([f.run, f.setting], f.auto_term, normalize="index").round(3).to_string())
    print("\ns_auto / s_log per programming, lambda >= 2 lambda_exact: quantiles by run, n, setting")
    g2 = f[f.region == "ge2"]
    print(g2.groupby(["run", "n", "setting"]).ratio.quantile([0.05, 0.5, 0.95]).unstack().round(3).to_string())
    print("\nchain couplers and strongest problem coupler in hardware units (median, lambda >= 2 lambda_exact):")
    print(g2.groupby(["run", "n", "setting"])[["chain_hw", "jprob_hw"]].median().round(3).to_string())
    c1 = g2[g2.run == "c1"]
    print("\nc1, lambda >= 2 lambda_exact: strongest problem coupler in hardware units by setting (quantiles)")
    print(c1.groupby(["n", "setting"]).jprob_hw.quantile([0.05, 0.5, 0.95]).unstack().round(3).to_string())

    predict = master_curve("report/scaling_points.csv")
    rng = np.random.default_rng(0)
    d = g2[(g2.run != "c1") & g2.gap.notna()]
    rows = []
    for (run, n), g in list(d.groupby(["run", "n"])) + [(("main pooled", 0), d[d.run == "main"])]:
        ids = g.inst_id.unique()
        for col in ["s_log", "s_auto"]:
            b, r2 = best_beta(g, col, predict)
            bs = [] if run == "main pooled" else [
                best_beta(pd.concat([g[g.inst_id == i] for i in rng.choice(ids, len(ids))]), col, predict)[0]
                for _ in range(args.boot)]
            rows.append(dict(run=run, n=n, chain_len=round(g.chain_len.mean(), 2), scale=col, beta=b, r2=round(r2, 3),
                             lo=np.quantile(bs, 0.025) if bs else np.nan, hi=np.quantile(bs, 0.975) if bs else np.nan,
                             points=len(g)))
    res = pd.DataFrame(rows)
    res.to_csv("embedded_gauge_beta.csv", index=False)
    print("\nscaling fit per programming (one point per half), s_log vs gauge-aware s_auto:")
    print(res.pivot_table(index=["run", "n", "chain_len"], columns="scale", values=["beta", "lo", "hi", "r2"])
          .round(3).to_string())


if __name__ == "__main__":
    main()
