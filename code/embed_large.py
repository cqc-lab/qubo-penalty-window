"""Precompute the minorminer embeddings of the 60 large sparse instances for one device, in parallel.

Uses the same call as qpu_large.embedding_for (interaction graph of |U| + |P|, random_seed = instance seed,
tries=20, timeout=120) and writes them into the embedding cache under the key "<chip>:sparse_<n>_<M>_<idx>", so
qpu_large.py and qpu_budget.py find them. The device graph is a JSON file with an "edges" list (fetch_graph.py).

    python embed_large.py --graph graph_Advantage2_system1.json --chip Advantage2_system1 --workers 30
"""
import argparse
import json
import os
from multiprocessing import Pool

import minorminer
import networkx as nx
import numpy as np
import pandas as pd

import large

G = None


def embed(task):
    n, M, idx = task
    inst, seed = large.make_large("sparse", n, M, idx)
    W = np.abs(inst["U"]) + np.abs(inst["P"])
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if W[i, j] != 0]
    emb = minorminer.find_embedding(edges, G.edges, random_seed=seed, tries=20, timeout=120)
    return task, {str(v): list(map(int, emb[v])) for v in range(n)} if emb else None


def main():
    global G
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--chip", required=True)
    ap.add_argument("--ref", default="large.csv")
    ap.add_argument("--cache", default="qpu_large_embeddings.json")
    ap.add_argument("--workers", type=int, default=30)
    args = ap.parse_args()
    G = nx.Graph()
    G.add_edges_from(map(tuple, json.load(open(args.graph))["edges"]))
    cache = json.load(open(args.cache)) if os.path.exists(args.cache) else {}
    refs = pd.read_csv(args.ref).drop_duplicates(["family", "n", "M", "idx"])
    tasks = [(int(r.n), int(r.M), int(r.idx)) for r in refs[refs.family == "sparse"].itertuples()
             if f"{args.chip}:sparse_{int(r.n)}_{int(r.M)}_{int(r.idx)}" not in cache]
    with Pool(args.workers) as pool:
        res = pool.map(embed, tasks)
    for (n, M, idx), emb in res:
        if emb is None:
            print(f"no embedding for n={n} M={M} #{idx}")
            continue
        cache[f"{args.chip}:sparse_{n}_{M}_{idx}"] = emb
        print(f"n={n} M={M} #{idx}: mean chain {np.mean([len(c) for c in emb.values()]):.2f}, "
              f"max {max(len(c) for c in emb.values())}")
    json.dump(cache, open(args.cache, "w"))


if __name__ == "__main__":
    main()
