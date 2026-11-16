"""Hardware scale of the embedded problem under the QPU's actual auto_scale rule.

embedded_scale.py assumed the symmetric ranges h in [-4, 4], J in [-1, 1]. Since 2023-11-15 the Leap auto_scale
parameter (on by default) uses extended_j_range for J and, on Advantage systems, also bounds the total coupling
per qubit (per_qubit_coupling_range). The factor the QPU divides the embedded problem by is

    max( max(h)/h_max, min(h)/h_min, max(J)/J_max_ext, min(J)/J_min_ext,
         max(sum_J_qubit)/pq_max, min(sum_J_qubit)/pq_min ),   each term clipped at 0,

with h_range [-4, 4], extended_j_range [-2, 1] and per_qubit_coupling_range [-18, 15] (Advantage_system6 solver
properties, qpu_solver_properties.json; Advantage_system4, now retired, belonged to the same generation and is
assumed to have had the same ranges). This script recomputes the factor for every QPU point of the main, chain,
large-n and chain-strength (qpu_c1.csv) runs, reports which term sets it, the ratio to the logical scale s(lambda),
the strength of the strongest problem coupler in hardware units for the two chain-strength settings, and refits
beta_eff by the scaling fit with this scale.

    python embedded_autoscale.py --workers 30
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

# Full device graph if present (fetch_graph.py), otherwise the published couplers of the embeddings.
GRAPH = next(p for p in ["graph_Advantage_system4.json", "embedding_couplers_Advantage_system4.json"]
             if os.path.exists(p))
from embedded_scale import SEED, best_beta, master_curve

H_RANGE = (-4.0, 4.0)
J_EXT = (-2.0, 1.0)
PER_QUBIT = (-18.0, 15.0)
RUNS = {"main": ("qpu_main.csv", {}), "chain": ("qpu_chain.csv", {10: "10of14", 18: "18of30"}),
        "bign": ("qpu_bign.csv", {}), "c1": ("qpu_c1.csv", {})}
TERMS = ["h", "J+", "J-", "qubit+", "qubit-"]


def load_adj(path):
    adj = {}
    for u, v in json.load(open(path))["edges"]:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj


def autoscale(tb):
    h = np.fromiter(tb.linear.values(), float)
    J = np.fromiter(tb.quadratic.values(), float)
    tot = dict.fromkeys(tb.linear, 0.0)
    for (u, v), b in tb.quadratic.items():
        tot[u] += b
        tot[v] += b
    t = np.fromiter(tot.values(), float)
    terms = [max(h.max() / H_RANGE[1], h.min() / H_RANGE[0], 0.0), max(J.max() / J_EXT[1], 0.0),
             max(J.min() / J_EXT[0], 0.0), max(t.max() / PER_QUBIT[1], 0.0), max(t.min() / PER_QUBIT[0], 0.0)]
    return max(terms), TERMS[int(np.argmax(terms))]


def features(task):
    run, inst_id, points, emb_key = task
    n, M, idx = (int(t[1:]) if i < 2 else int(t) for i, t in enumerate(inst_id.split("_")))
    seed = int(np.random.SeedSequence([SEED, n, M, idx]).generate_state(1)[0])
    inst = gonogo.make_instance(n, M, seed)
    fe = np.unique(inst["E"][inst["feas"]])
    gap = float(fe[1] - fe[0]) if len(fe) > 1 else np.nan
    adj = load_adj(GRAPH)
    emb = {int(v): c for v, c in json.load(open("qpu_embeddings.json"))[emb_key].items()}
    chains = [set(c) for c in emb.values()]
    rows = []
    for ll, setting in points:
        h, J = gonogo.qubo_to_ising(inst["U"] + 10.0 ** ll * inst["P"])
        s_log = max(np.abs(h).max() / gonogo.H_RANGE, np.abs(J).max() / gonogo.J_RANGE)
        bqm = dimod.BinaryQuadraticModel({i: h[i] for i in range(n)},
                                         {(i, j): J[i, j] for i in range(n) for j in range(i + 1, n) if J[i, j]},
                                         0.0, "SPIN")
        cs = s_log if setting == "scaled" else uniform_torque_compensation(bqm, emb)
        tb = embed_bqm(bqm, emb, adj, chain_strength=cs)
        s_auto, term = autoscale(tb)
        hl = np.abs(np.fromiter(tb.linear.values(), float))
        jq = np.abs(np.fromiter(tb.quadratic.values(), float))
        s_sym = max(hl.max() / 4.0, jq.max() / 1.0)
        jprob = max(abs(b) for (u, v), b in tb.quadratic.items() if not any(u in c and v in c for c in chains))
        rows.append(dict(run=run, inst_id=inst_id, n=n, loglam=ll, setting=setting, lam_exact=inst["lam_exact"],
                         gap=gap, chain_len=np.mean([len(c) for c in emb.values()]), chain_strength=cs, s_log=s_log,
                         s_sym=s_sym, s_auto=s_auto, auto_term=term, jprob_hw=jprob / s_auto,
                         chain_hw=cs / s_auto))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--out", default="embedded_autoscale.csv")
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()

    if not os.path.exists(args.out):
        tasks = []
        for run, (path, keys) in RUNS.items():
            q = pd.read_csv(path)
            if "setting" not in q.columns:
                q["setting"] = "default"
            for iid, g in q.groupby("inst_id"):
                n = int(iid.split("_")[0][1:])
                pts = sorted(set(zip(g.loglam.round(6), g.setting)))
                tasks.append((run, iid, pts, keys.get(n, str(n))))
        with Pool(args.workers) as pool:
            pd.DataFrame([r for rs in pool.map(features, tasks) for r in rs]).to_csv(args.out, index=False)
    f = pd.read_csv(args.out)
    rel = f.loglam - np.log10(f.lam_exact.where(f.lam_exact > 0))
    f["region"] = np.where(rel >= np.log10(2), "ge2", np.where(rel.abs() <= 0.25, "near", "other"))
    pd.set_option("display.width", 200)

    print("term that sets the hardware scale (fraction of points), by run and setting:")
    print(pd.crosstab([f.run, f.setting], f.auto_term, normalize="index").round(3).to_string())
    print("\nmedian s_auto / s_log and s_sym / s_log, by run, n, setting and region:")
    print(f.assign(r_auto=f.s_auto / f.s_log, r_sym=f.s_sym / f.s_log)
          .groupby(["run", "n", "setting", "region"])[["r_auto", "r_sym", "chain_hw"]].median().round(3).to_string())

    c1 = f[f.run == "c1"].pivot_table(index=["inst_id", "loglam"], columns="setting", values="jprob_hw")
    c1["ratio"] = c1.scaled / c1.default
    c1 = c1.join(f[f.run == "c1"].drop_duplicates(["inst_id", "loglam"]).set_index(["inst_id", "loglam"])[["n", "region"]])
    print("\nstrongest problem coupler in hardware units, scaled / default (median by n, region):")
    print(c1.groupby(["n", "region"]).ratio.median().round(3).to_string())
    print(c1.groupby("n").ratio.describe().round(3).to_string())

    # scaling fit with the auto-scaled embedded problem (default chain strength runs)
    q = pd.concat([pd.read_csv(p).assign(run=r) for r, (p, _) in RUNS.items() if r != "c1"])
    q = q.drop(columns=["chain_len", "n", "lam_exact"], errors="ignore")
    q["success"] = (q.succ_a + q.succ_b) / (q.reads_a + q.reads_b)
    q["loglam"] = q.loglam.round(6)
    d = q.merge(f[f.run != "c1"], on=["run", "inst_id", "loglam"])
    d = d[(d.region == "ge2") & d.gap.notna()]
    predict = master_curve("report/scaling_points.csv")
    rng = np.random.default_rng(0)
    rows = []
    groups = list(d.groupby(["run", "n"])) + [(("main pooled", 0), d[d.run == "main"])]
    for (run, n), g in groups:
        ids = g.inst_id.unique()
        for col in ["s_log", "s_sym", "s_auto"]:
            b, r2 = best_beta(g, col, predict)
            bs = [] if run == "main pooled" else [
                best_beta(pd.concat([g[g.inst_id == i] for i in rng.choice(ids, len(ids))]), col, predict)[0]
                for _ in range(args.boot)]
            rows.append(dict(run=run, n=n, chain_len=round(g.chain_len.mean(), 2), scale=col, beta=b, r2=round(r2, 3),
                             lo=np.quantile(bs, 0.025) if bs else np.nan, hi=np.quantile(bs, 0.975) if bs else np.nan,
                             points=len(g)))
    res = pd.DataFrame(rows)
    res.to_csv("embedded_autoscale_beta.csv", index=False)
    print("\nscaling-fit beta_eff by scale (s_log logical, s_sym old symmetric-range embedded, s_auto actual):")
    print(res.pivot_table(index=["run", "n", "chain_len"], columns="scale", values=["beta", "lo", "hi", "r2"]).round(3)
          .to_string())


if __name__ == "__main__":
    main()
