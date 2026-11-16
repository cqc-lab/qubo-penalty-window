"""Reduce a device coupler graph to the couplers that the study's embeddings use.

The full working graph of a D-Wave system is not redistributed. The embedded-scale analyses
(embedded_autoscale.py, embedded_scale.py, c1_analyze.py) only need the couplers among the qubits of each
embedding, because embed_bqm places chain and problem couplings only there. This script keeps exactly those:
the union, over all embeddings, of the couplers whose two qubits belong to the same embedding. The output has
the same format as fetch_graph.py, so the analyses run on it unchanged.

    python used_couplers.py graph_Advantage_system4.json qpu_embeddings.json embedding_couplers_Advantage_system4.json
"""
import json
import sys


def main():
    graph_path, emb_path, out_path = sys.argv[1:4]
    g = json.load(open(graph_path))
    embs = json.load(open(emb_path))
    keep = set()
    for chains in embs.values():
        qubits = {q for c in chains.values() for q in c}
        keep |= {(u, v) for u, v in g["edges"] if u in qubits and v in qubits}
    json.dump({"edges": sorted(map(list, keep)), "h_range": g["h_range"], "j_range": g["j_range"],
               "note": "couplers among the qubits of each embedding in " + emb_path + "; not the full device graph"},
              open(out_path, "w"))
    print(f"{out_path}: {len(keep)} of {len(g['edges'])} couplers, {len(embs)} embeddings")


if __name__ == "__main__":
    main()
