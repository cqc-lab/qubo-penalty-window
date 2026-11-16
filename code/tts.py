"""Time-to-solution view of the selection results, charging every method for its selection reads.

TTS99 = reads needed to see the optimum at least once with probability 0.99, at the per-read success p of the
chosen lambda. total = selection reads + TTS99. P(N) = probability of seeing the optimum within a total budget
of N reads (selection reads included).

    python tts.py estimator.csv qpu_estimator.csv qpu_garcia.csv
"""
import sys

import numpy as np
import pandas as pd

ROUND_READS = 250
SELECT = {"est": 1000, "est_nomargin": 1000, "seq": 1000, "seq15": 1000, "grid": 1000, "est_re": 1000,
          "oracle": 0, "inst": 0, "wc": 0}


def selection_reads(row):
    if row.method in SELECT:
        return SELECT[row.method]
    return row.rounds * ROUND_READS  # Garcia et al. variants: one round = 250 reads


def summarize(df, budgets=(2000, 5000)):
    df = df.copy()
    df["sel"] = df.apply(selection_reads, axis=1)
    p = df.success.clip(0, 1)
    with np.errstate(divide="ignore"):
        tts = np.where(p >= 0.99, 1, np.ceil(np.log(0.01) / np.log1p(-p.clip(upper=0.99))))
    df["tts"] = np.where(p > 0, tts, np.inf)
    df["total"] = df.sel + df.tts
    for N in budgets:
        df[f"P{N}"] = 1 - (1 - p) ** np.maximum(N - df.sel, 0)
    g = df.groupby("method")
    out = pd.DataFrame({"success": g.success.mean(), "sel_reads": g.sel.mean(),
                        "median_total": g.total.median(), "q90_total": g.total.quantile(0.9)})
    for N in budgets:
        out[f"P(N={N})"] = g[f"P{N}"].mean()
    # the hardest quarter of instances (lowest oracle success), where the choice of lambda matters most
    ref = "oracle" if "oracle" in df.method.values else "est_re"
    hard = df[df.method == ref].nsmallest(max(1, df.inst_id.nunique() // 4), "success").inst_id
    out["median_total_hard"] = df[df.inst_id.isin(hard)].groupby("method").total.median()
    return out.sort_values("median_total")


def main():
    for path in sys.argv[1:]:
        df = pd.read_csv(path)
        if "cond" in df and df.cond.nunique() > 1:
            df = df[df.cond == df.cond.iloc[0]]
        print(f"\n== {path}: {df.inst_id.nunique()} instances")
        print(summarize(df).round(3).to_string())


if __name__ == "__main__":
    main()
