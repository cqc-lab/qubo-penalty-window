"""Save the coupler graph and h/J ranges of a D-Wave solver (needed by embedded_scale.py).

Reading solver properties uses no QPU access time, but needs a configured Ocean token.

    python fetch_graph.py Advantage_system4
"""
import json
import sys

from dwave.cloud import Client


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "Advantage_system4"
    with Client.from_config() as client:
        s = client.get_solver(name=name)
        json.dump({"edges": sorted(map(list, s.edges)), "h_range": s.properties["h_range"],
                   "j_range": s.properties["j_range"]}, open(f"graph_{name}.json", "w"))
    print(f"graph_{name}.json: {len(s.edges)} couplers")


if __name__ == "__main__":
    main()
