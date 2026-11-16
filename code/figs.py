"""Figure 1: success rate (top) and feasible fraction (bottom) against lambda / lambda_exact for software SA (C0),
emulated QPU (C3) and the QPU.

    python figs.py [outdir]          # reads results.csv and qpu_main.csv from the working directory

The beta_eff figure is made by fig_beta.py.
"""
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures"
GRID, INK, MUTED = "#e4e3dd", "#2b2a27", "#898781"


def style(ax):
    ax.grid(color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


os.makedirs(OUT, exist_ok=True)
r = pd.read_csv("results.csv")
r = r[r.cond.isin(["C0_raw", "C3_ice02"]) & ((r.M == (r.n * 0.5).round().astype(int)) | (r.M == r.n))]
r["s"] = (r.succ_a + r.succ_b) / (r.reads_a + r.reads_b)
r["f"] = (r.feas_a + r.feas_b) / (r.reads_a + r.reads_b)
q = pd.read_csv("qpu_main.csv")
q["s"] = (q.succ_a + q.succ_b) / (q.reads_a + q.reads_b)
q["f"] = (q.feas_a + q.feas_b) / (q.reads_a + q.reads_b)
d = pd.concat([r[["n", "cond", "loglam", "lam_exact", "s", "f"]], q[["n", "cond", "loglam", "lam_exact", "s", "f"]]])
d = d[d.lam_exact > 0]
d["rel"] = d.loglam - np.log10(d.lam_exact)
d["bin"] = (d.rel / 0.25).round() * 0.25
fig, axs = plt.subplots(2, 3, figsize=(12, 6.4), sharey="row", sharex=True)
spec = {"C0_raw": ("#898781", "SA, auto β (original setting)"), "C3_ice02": ("#2a78d6", "emulated QPU (β=5, σ=0.02)"),
        "QPU": ("#e34948", "D-Wave Advantage_system4")}
for row, (var, ylab) in enumerate([("s", "global-optimum success rate"), ("f", "feasible fraction of reads")]):
    for ax, n in zip(axs[row], [10, 14, 18]):
        for c, (col, lab) in spec.items():
            g = d[(d.n == n) & (d.cond == c)]
            m = g.groupby("bin")[var].agg(["mean", "count"])
            m = m[(m["count"] >= 8) & (m.index <= 3.0) & (m.index >= -1)]
            ax.plot(m.index, m["mean"], color=col, lw=2, marker="o" if c == "QPU" else None, ms=5, mec="white",
                    mew=1, label=lab)
        ax.axvline(np.log10(1.5), color=MUTED, lw=1, ls="--")
        if row == 0:
            ax.set_title(f"n = {n}", fontsize=10, color=INK)
        else:
            ax.set_xlabel(r"$\log_{10}(\lambda/\lambda_{\mathrm{exact}})$")
        style(ax)
    axs[row, 0].set_ylabel(ylab)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(os.path.join(OUT, "window_collapse.png"), dpi=200)
