"""Scaling law with the auto-scale factor of the embedded problem, and bootstrap intervals for beta_eff.

The QPU runs used Ocean's default chain strength (uniform torque compensation), which grows with lambda.
The scale the hardware actually applies is that of the embedded problem, chain couplers included, not the
logical s(lambda) used so far. This script recomputes it for every QPU point and refits beta_eff by the
scaling fit (SA master curve, predictive R^2) with either scale, per run and per n, with bootstrap
intervals over instances.

    python embedded_scale.py --workers 30
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

SEED = 2026
RUNS = {"main": ("qpu_main.csv", {}), "chain": ("qpu_chain.csv", {10: "10of14", 18: "18of30"}),
        "bign": ("qpu_bign.csv", {})}
BETAS = np.round(np.arange(0.5, 15.0001, 0.05), 2)


def load_graph(path):
    g = json.load(open(path))
    adj = {}
    for u, v in g["edges"]:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj, g["h_range"][1], g["j_range"][1]


def features(task):
    run, inst_id, loglams, emb_key = task
    n, M, idx = (int(t[1:]) if i < 2 else int(t) for i, t in enumerate(inst_id.split("_")))
    seed = int(np.random.SeedSequence([SEED, n, M, idx]).generate_state(1)[0])
    inst = gonogo.make_instance(n, M, seed)
    fe = np.unique(inst["E"][inst["feas"]])
    gap = float(fe[1] - fe[0]) if len(fe) > 1 else np.nan
    adj, h_max, j_max = load_graph(GRAPH)
    emb = {int(v): c for v, c in json.load(open("qpu_embeddings.json"))[emb_key].items()}
    rows = []
    for ll in loglams:
        h, J = gonogo.qubo_to_ising(inst["U"] + 10.0 ** ll * inst["P"])
        s_log = max(np.abs(h).max() / gonogo.H_RANGE, np.abs(J).max() / gonogo.J_RANGE)
        bqm = dimod.BinaryQuadraticModel({i: h[i] for i in range(n)},
                                         {(i, j): J[i, j] for i in range(n) for j in range(i + 1, n) if J[i, j]},
                                         0.0, "SPIN")
        cs = uniform_torque_compensation(bqm, emb)
        tb = embed_bqm(bqm, emb, adj, chain_strength=cs)
        hl = np.abs(np.fromiter(tb.linear.values(), float))
        jq = np.abs(np.fromiter(tb.quadratic.values(), float))
        jq_prob = max((abs(b) for (u, v), b in tb.quadratic.items()
                       if not any(u in c and v in c for c in emb.values())), default=0.0)
        s_emb = max(hl.max() / h_max, jq.max() / j_max)
        rows.append(dict(run=run, inst_id=inst_id, n=n, loglam=ll, chain_len=np.mean([len(c) for c in emb.values()]), lam_exact=inst["lam_exact"], gap=gap,
                         s_log=s_log, s_emb=s_emb, chain_strength=cs, chain_sets_scale=cs / j_max >= s_emb - 1e-12,
                         s_emb_problem=max(hl.max() / h_max, jq_prob / j_max)))
    return rows


def master_curve(path, width=0.1):
    sa = pd.read_csv(path)
    sa = sa[sa.cond.isin(["S1_beta2", "C3_ice02", "S2_beta10"]) & (sa.x_rel >= np.log10(2)) & sa.gap.notna()]
    m = sa.groupby(np.floor(sa.x_therm / width)).r.mean()
    return lambda x: np.interp(np.floor(x / width), m.index.to_numpy(), m.to_numpy())


def best_beta(d, scale_col, predict):
    y = d.success.to_numpy()
    sst = np.sum((y - y.mean()) ** 2)
    r2 = np.array([1 - np.sum((y - predict(np.log10(b * d.gap.to_numpy() / d[scale_col].to_numpy()))) ** 2) / sst
                   for b in BETAS])
    return BETAS[r2.argmax()], r2.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--out", default="embedded_scale.csv")
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()

    if not os.path.exists(args.out):
        tasks = []
        for run, (path, keys) in RUNS.items():
            q = pd.read_csv(path)
            for iid, g in q.groupby("inst_id"):
                n = int(iid.split("_")[0][1:])
                tasks.append((run, iid, sorted(g.loglam.unique()), keys.get(n, str(n))))
        with Pool(args.workers) as pool:
            pd.DataFrame([r for rs in pool.map(features, tasks) for r in rs]).to_csv(args.out, index=False)
    f = pd.read_csv(args.out)
    q = pd.concat([pd.read_csv(p).assign(run=r) for r, (p, _) in RUNS.items()]).drop(columns=["chain_len"], errors="ignore")
    q["success"] = (q.succ_a + q.succ_b) / (q.reads_a + q.reads_b)
    d = q.drop(columns=["n", "lam_exact"]).merge(f, on=["run", "inst_id", "loglam"])
    d = d[(d.lam_exact > 0) & (d.loglam - np.log10(d.lam_exact.where(d.lam_exact > 0)) >= np.log10(2))
          & d.gap.notna()]
    print(f"{len(d)} QPU points at lambda >= 2 lambda_exact")
    print("\nembedded / logical scale (median by run, n):")
    print(d.assign(ratio=d.s_emb / d.s_log).groupby(["run", "n"]).agg(
        ratio=("ratio", "median"), chain_sets_scale=("chain_sets_scale", "mean"),
        chain_len=("chain_len", "mean")).round(3).to_string())

    predict = master_curve("report/scaling_points.csv")
    rng = np.random.default_rng(0)
    rows = []
    for (run, n), g in d.groupby(["run", "n"]):
        ids = g.inst_id.unique()
        for col in ["s_log", "s_emb"]:
            b, r2 = best_beta(g, col, predict)
            bs = []
            for _ in range(args.boot):
                pick = rng.choice(ids, len(ids))
                gg = pd.concat([g[g.inst_id == i] for i in pick])
                bs.append(best_beta(gg, col, predict)[0])
            rows.append(dict(run=run, n=n, chain_len=round(g.chain_len.mean(), 2), scale=col, beta=b, r2=round(r2, 3),
                             lo=np.quantile(bs, 0.025), hi=np.quantile(bs, 0.975), points=len(g)))
    for col in ["s_log", "s_emb"]:
        g = d[d.run == "main"]
        b, r2 = best_beta(g, col, predict)
        rows.append(dict(run="main pooled", n=0, chain_len=np.nan, scale=col, beta=b, r2=round(r2, 3),
                         lo=np.nan, hi=np.nan, points=len(g)))
    res = pd.DataFrame(rows)
    res.to_csv("embedded_beta.csv", index=False)
    print("\nscaling-fit beta_eff with 95% bootstrap intervals over instances:")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
