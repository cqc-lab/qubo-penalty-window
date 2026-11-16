"""Figure: beta_eff against mean chain length, with 95% bootstrap intervals.

Advantage_system4: fine direct fit and scaling fit (beta_fine.csv, embedded_beta.csv). Optional Advantage_system6
points: the interleaved chain-length run (interleave_beta.csv, direct fit) and the large instances
(qpu_large_summary.csv, near5 curves matched to emulation on a coarse beta grid). The dashed line is 7 / chain length.

    python fig_beta.py beta_fine.csv embedded_beta.csv figures/beta_chain.png [interleave_beta.csv qpu_large_summary.csv]
"""
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11})

GRID, MUTED = "#e4e3dd", "#898781"
CHAIN = {("main", 10): 2.0, ("main", 14): 3.0, ("main", 18): 3.0, ("chain", 10): 3.0, ("chain", 18): 3.89,
         ("bign", 20): 3.0, ("bign", 22): 3.82}
OFFSET = {10: -0.09, 14: -0.03, 18: 0.03, 20: 0.09, 22: 0.0}


def main():
    direct = pd.read_csv(sys.argv[1])
    scal = pd.read_csv(sys.argv[2])
    scal = scal[(scal.scale == "s_log") & (scal.run != "main pooled")]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for df, col, mk, lab, dx in [(direct, "#2a78d6", "o", "direct fit, system4", 0.0),
                                 (scal, "#eb6834", "s", "scaling fit, system4", 0.02)]:
        x = np.array([CHAIN[(r, n)] + OFFSET[n] + dx for r, n in zip(df.run, df.n)])
        ax.errorbar(x, df.beta, yerr=[df.beta - df.lo, df.hi - df.beta], fmt=mk, color=col, ms=7, mec="white",
                    mew=1, elinewidth=1.2, capsize=0, label=lab, zorder=3)
    if len(sys.argv) > 5:
        il = pd.read_csv(sys.argv[4])
        il = il[il.emb.isin(["short", "long"])]
        ax.errorbar(il.chain_len + 0.06, il.beta, yerr=[il.beta - il.lo, il.hi - il.beta], fmt="^", color="#1baf7a",
                    ms=7, mec="white", mew=1, elinewidth=1.2, capsize=0, label="direct fit, system6 (interleaved)",
                    zorder=3)
        lg = pd.read_csv(sys.argv[5])
        lg = lg[lg.part == "beta"]
        for fam, mk in [("sparse", "D"), ("dense", "P")]:
            d = lg[lg.family == fam]
            ax.errorbar(d.chain_len, d.value, yerr=[d.value - d.lo, d.hi - d.value], fmt=mk, color="#8a5cd6", ms=7,
                        mec="white", mew=1, elinewidth=1.2, capsize=0, zorder=3,
                        label=f"near5 match, system6 ({fam}, $n$={'50-100' if fam == 'sparse' else '50'})")
        xs = np.linspace(1.8, 6.1, 100)
        ax.plot(xs, 7 / xs, color=MUTED, ls="--", lw=1, zorder=1, label="7 / chain length")
        ax.set_xticks([2, 3, 4, 5, 6])
        ax.set_xlim(1.7, 6.2)
    else:
        ax.set_xticks([2, 3, 4])
        ax.set_xlim(1.7, 4.3)
    ax.set_xlabel("mean chain length")
    ax.set_ylabel(r"$\beta_{\mathrm{eff}}$")
    ax.grid(color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=9.5, frameon=False, loc="upper right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(sys.argv[3]) or ".", exist_ok=True)
    fig.savefig(sys.argv[3], dpi=200)


if __name__ == "__main__":
    main()
