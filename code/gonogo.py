"""Lambda sweep: does the penalty window narrow under QPU-like scaling, temperature and ICE noise?

Sampling conditions are in CONDITIONS (paper, Table 1). The study began as a pilot ("go/no-go") check, and
`analyze` still prints that pilot's decision rule (verdict()); the paper does not use it.

    python gonogo.py run --out results.csv [--quick] [--workers 6]
    python gonogo.py analyze --csv results.csv --outdir report
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import dimod

try:
    from dwave.samplers import SimulatedAnnealingSampler
except ImportError:
    from neal import SimulatedAnnealingSampler

B = 10                       # objective coefficients drawn from [-B, B]
H_RANGE, J_RANGE = 4.0, 1.0  # D-Wave Advantage auto-scale ranges
FORBID = {0: (0, 1), 1: (0, 0), 2: (1, 1), 3: (1, 0)}  # constraint type -> forbidden (x_i, x_j)

# name: (auto-scale to QPU ranges, SA beta_range in scaled units or None for neal's auto, ICE sigma)
CONDITIONS = {
    "C0_raw": (False, None, 0.0),
    "C1_scaled": (True, (0.1, 5.0), 0.0),
    "C2_ice01": (True, (0.1, 5.0), 0.01),
    "C3_ice02": (True, (0.1, 5.0), 0.02),
    "C4_ice05": (True, (0.1, 5.0), 0.05),
    "S1_beta2": (True, (0.1, 2.0), 0.02),
    "S2_beta10": (True, (0.1, 10.0), 0.02),
}
MAIN = "C3_ice02"

FULL = dict(ns=[10, 14, 18], rhos=[0.25, 0.5, 1.0], n_inst=50,
            loglams=np.round(np.arange(-0.5, 4.5 + 1e-9, 0.25), 3),
            n_prog=4, reads=125, sweeps=1000)
# Same instances as FULL, dense grid over the window seen in the FULL run, 4x the reads.
FINE = dict(FULL, loglams=np.round(np.arange(0.5, 2.5 + 1e-9, 0.05), 3), n_prog=8, reads=250)
QUICK = dict(ns=[8, 10], rhos=[0.5, 1.0], n_inst=4,
             loglams=np.round(np.arange(-0.5, 4.5 + 1e-9, 0.5), 3),
             n_prog=2, reads=50, sweeps=500)


# ---------------------------------------------------------------- instances

def all_states(n):
    return ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(np.int8)


def make_instance(n, M, seed):
    """Random QUBO objective + M satisfiable 2-SAT constraints, solved by brute force."""
    rng = np.random.default_rng(seed)
    X = all_states(n)
    Q = rng.integers(-B, B + 1, size=(n, n)).astype(float)
    Xf = X.astype(float)
    E = ((Xf @ Q) * Xf).sum(1)

    V = np.zeros(len(X), np.int32)  # number of violated constraints per state
    cons = []
    tries = 0
    while len(cons) < M:
        tries += 1
        if tries > 1000 * max(M, 1):
            raise RuntimeError(f"could not place {M} satisfiable constraints for n={n}")
        i, j = rng.choice(n, 2, replace=False)
        t = int(rng.integers(4))
        fi, fj = FORBID[t]
        viol = (X[:, i] == fi) & (X[:, j] == fj)
        if ((V == 0) & ~viol).any():
            V += viol
            cons.append((int(i), int(j), t))

    feas = V == 0
    E_star = E[feas].min()
    infeas = ~feas
    lam_exact = 0.0
    if infeas.any():
        lam_exact = max(0.0, float(((E_star - E[infeas]) / V[infeas]).max()))
    Qs = Q + Q.T
    delta = np.abs(np.diag(Q)) + np.abs(Qs).sum(1) - np.abs(np.diag(Qs))
    lam_inst = float(delta.max())

    # Upper-triangular QUBO for the objective (diag = linear) and for the penalty sum.
    U = np.triu(Qs, 1) + np.diag(np.diag(Q))
    P = np.zeros((n, n))
    for i, j, t in cons:
        fi, fj = FORBID[t]
        a, b = (0.0, 1.0) if fi else (1.0, -1.0)  # literal l_f(x) = a + b x
        c, d = (0.0, 1.0) if fj else (1.0, -1.0)
        P[i, i] += b * c
        P[j, j] += a * d
        P[min(i, j), max(i, j)] += b * d
    return dict(U=U, P=P, feas=feas, E=E, E_star=E_star,
                lam_exact=lam_exact, lam_inst=lam_inst, lam_wc=float((2 * n - 1) * B))


def qubo_to_ising(W):
    """Upper-triangular QUBO (diag = linear) -> (h, J upper-triangular), constant dropped."""
    Qu = np.triu(W, 1)
    h = np.diag(W) / 2 + (Qu.sum(1) + Qu.sum(0)) / 4
    return h, Qu / 4


# ---------------------------------------------------------------- sampling

def sample_condition(h, J, cond, cfg, rng, sampler):
    """Returns a list of (n_prog) spin arrays, one per programming, columns in variable order."""
    scale, beta, sigma = CONDITIONS[cond]
    if scale:
        s = max(np.abs(h).max() / H_RANGE, np.abs(J).max() / J_RANGE)
        h, J = h / s, J / s
    mask = J != 0
    out = []
    for _ in range(cfg["n_prog"]):
        hp, Jp = h, J
        if sigma > 0:
            hp = h + rng.normal(0.0, sigma, h.shape)
            Jp = J + rng.normal(0.0, sigma, J.shape) * mask
        bqm = dimod.BinaryQuadraticModel(hp, Jp, 0.0, "SPIN")
        ss = sampler.sample(bqm, num_reads=cfg["reads"], num_sweeps=cfg["sweeps"],
                            beta_range=beta, seed=int(rng.integers(2**31 - 1)))
        order = np.argsort(np.asarray(list(ss.variables)))
        out.append(ss.record.sample[:, order])
    return out


def run_instance(task):
    n, M, idx, seed, cfg = task
    inst = make_instance(n, M, seed)
    rng = np.random.default_rng(seed + 1)
    sampler = SimulatedAnnealingSampler()
    weights = 1 << np.arange(n)
    rows = []
    for loglam in cfg["loglams"]:
        h, J = qubo_to_ising(inst["U"] + 10.0 ** loglam * inst["P"])
        for cond in CONDITIONS:
            succ, feas = [0, 0], [0, 0]
            for p, S in enumerate(sample_condition(h, J, cond, cfg, rng, sampler)):
                k = ((S + 1) // 2) @ weights
                f = inst["feas"][k]
                feas[p % 2] += int(f.sum())
                succ[p % 2] += int((f & (inst["E"][k] == inst["E_star"])).sum())
            half = cfg["reads"] * (cfg["n_prog"] // 2)
            rows.append(dict(inst_id=f"n{n}_M{M}_{idx}", n=n, M=M, cond=cond, loglam=loglam,
                             succ_a=succ[0], succ_b=succ[1], feas_a=feas[0], feas_b=feas[1],
                             reads_a=half, reads_b=cfg["reads"] * cfg["n_prog"] - half,
                             lam_exact=inst["lam_exact"], lam_inst=inst["lam_inst"],
                             lam_wc=inst["lam_wc"]))
    return rows


def cmd_run(args):
    cfg = QUICK if args.quick else FINE if args.fine else FULL
    done = set()
    if os.path.exists(args.out):
        done = set(pd.read_csv(args.out, usecols=["inst_id"]).inst_id)
    tasks = []
    for n in cfg["ns"]:
        for rho in cfg["rhos"]:
            M = max(1, round(rho * n))
            for idx in range(cfg["n_inst"]):
                if f"n{n}_M{M}_{idx}" in done:
                    continue
                seed = int(np.random.SeedSequence([args.seed, n, M, idx]).generate_state(1)[0])
                tasks.append((n, M, idx, seed, cfg))
    print(f"{len(tasks)} instances to run ({len(done)} already done)")
    with Pool(args.workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(run_instance, tasks), 1):
            pd.DataFrame(rows).to_csv(args.out, mode="a", index=False,
                                      header=not os.path.exists(args.out))
            print(f"[{i}/{len(tasks)}] {rows[0]['inst_id']}", flush=True)


# ---------------------------------------------------------------- analysis

def window(loglam, r, alpha):
    """Contiguous run around the argmax where r >= alpha * max r. Returns (lo, hi, censored)."""
    k = int(np.argmax(r))
    ok = r >= alpha * r[k]
    lo = hi = k
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(r) - 1 and ok[hi + 1]:
        hi += 1
    return loglam[lo], loglam[hi], lo == 0 or hi == len(r) - 1


def window_interp(loglam, r, alpha, k=3):
    """Window on a k-point moving average, with crossings linearly interpolated between grid points.

    Returns width in decades, or NaN when the window touches the grid edge (censored)."""
    rs = np.convolve(np.pad(r, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    i = int(np.argmax(rs))
    thr = alpha * rs[i]
    lo, hi = i, i
    while lo > 0 and rs[lo - 1] >= thr:
        lo -= 1
    while hi < len(rs) - 1 and rs[hi + 1] >= thr:
        hi += 1
    if lo == 0 or hi == len(rs) - 1:
        return np.nan
    x_lo = np.interp(thr, [rs[lo - 1], rs[lo]], [loglam[lo - 1], loglam[lo]])
    x_hi = np.interp(thr, [rs[hi + 1], rs[hi]], [loglam[hi + 1], loglam[hi]])
    return x_hi - x_lo


def per_instance(g, alpha, min_peak):
    g = g.sort_values("loglam")
    x = g.loglam.to_numpy()
    ra = (g.succ_a / g.reads_a).to_numpy()
    rb = (g.succ_b / g.reads_b).to_numpy()
    r = ((g.succ_a + g.succ_b) / (g.reads_a + g.reads_b)).to_numpy()
    first = g.iloc[0]
    out = dict(n=first.n, M=first.M, peak=r.max(), unsolved=r.max() < min_peak)
    lo, hi, cens = window(x, r, alpha)
    out.update(win_lo=lo, win_hi=hi, width=hi - lo, censored=cens,
               width90_i=window_interp(x, r, 0.9), width50_i=window_interp(x, r, 0.5),
               argmax_rel=x[np.argmax(r)] - np.log10(first.lam_exact) if first.lam_exact > 0 else np.nan)

    # Cross-validated peak: choose lambda on one half of the reads, score it on the other.
    out["peak_cv"] = 0.5 * (rb[np.argmax(ra)] + ra[np.argmax(rb)])
    rules = dict(wc=np.log10(first.lam_wc), inst=np.log10(first.lam_inst),
                 oracle=np.log10(max(1.5 * first.lam_exact, 10.0 ** x[0])))
    for name, lr in rules.items():
        at_rule = 0.5 * (np.interp(lr, x, rb) + np.interp(lr, x, ra))
        out[f"regret_{name}"] = out["peak_cv"] - at_rule
    return pd.Series(out)


def verdict(summary):
    main = summary[(summary.cond == MAIN) & (summary.n == summary.n.max())]
    best_rule = main[["regret_wc", "regret_inst"]].min(axis=1)
    narrow = main.width_med <= 1.5
    if (narrow & (best_rule >= 0.05)).any():
        v = "GO"
    elif (narrow & (best_rule < 0.02)).any():
        v = "PARTIAL GO (window narrows, simple rule is enough)"
    elif (main.width_med >= 3).all():
        v = "NO-GO"
    else:
        v = "GRAY (check on real QPU)"
    return v, main


def plot_curves(df, inst, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
    cmap = dict(zip(CONDITIONS, colors), QPU="#e34948")
    conds = [c for c in cmap if c in set(df.cond)]
    ns = sorted(df.n.unique())
    df = df.assign(r=(df.succ_a + df.succ_b) / (df.reads_a + df.reads_b))

    for fname, xcol, xlabel in [("curves_by_cond.png", "loglam", r"$\log_{10}\lambda$"),
                                ("curves_normalized.png", "xnorm",
                                 r"$\log_{10}(\lambda / \lambda_{\mathrm{exact}})$")]:
        d = df
        if xcol == "xnorm":
            d = df[df.lam_exact > 0].copy()
            d["xnorm"] = np.round((d.loglam - np.log10(d.lam_exact)) * 4) / 4
        fig, axes = plt.subplots(1, len(ns), figsize=(4.2 * len(ns), 3.6), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, n in zip(axes, ns):
            for cond in conds:
                c = d[(d.n == n) & (d.cond == cond)].groupby(xcol).r.mean()
                ax.plot(c.index, c.values, color=cmap[cond], lw=2, marker="o", ms=3, label=cond)
            ax.set_title(f"n = {n}", fontsize=10)
            ax.set_xlabel(xlabel)
            ax.grid(color="#e4e3dd", lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axes[0].set_ylabel("mean global-optimum success rate")
        axes[-1].legend(fontsize=7, frameon=False, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, fname), dpi=150)
        plt.close(fig)


def cmd_analyze(args):
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)
    df["r"] = (df.succ_a + df.succ_b) / (df.reads_a + df.reads_b)
    df["f"] = (df.feas_a + df.feas_b) / (df.reads_a + df.reads_b)
    df.groupby(["cond", "n", "M", "loglam"])[["r", "f"]].mean().reset_index() \
        .to_csv(os.path.join(args.outdir, "curves.csv"), index=False)

    inst = df.groupby(["cond", "inst_id"]).apply(
        per_instance, alpha=args.alpha, min_peak=args.min_peak, include_groups=False).reset_index()
    inst.to_csv(os.path.join(args.outdir, "per_instance.csv"), index=False)

    solved = inst[~inst.unsolved.astype(bool)]
    summary = inst.groupby(["cond", "n", "M"]).agg(
        n_inst=("peak", "size"), frac_unsolved=("unsolved", "mean")).join(
        solved.groupby(["cond", "n", "M"]).agg(
            width_med=("width", "median"), frac_censored=("censored", "mean"),
            w90_med=("width90_i", "median"), w90_q75=("width90_i", lambda s: s.quantile(0.75)),
            w50_med=("width50_i", "median"), w50_q75=("width50_i", lambda s: s.quantile(0.75)),
            peak_rel_med=("argmax_rel", "median"),
            peak_cv=("peak_cv", "mean"), regret_wc=("regret_wc", "mean"),
            regret_inst=("regret_inst", "mean"), regret_oracle=("regret_oracle", "mean"))
    ).reset_index()
    summary.to_csv(os.path.join(args.outdir, "summary.csv"), index=False)
    plot_curves(df, inst, args.outdir)

    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))
    if MAIN not in set(summary.cond):
        return
    v, main = verdict(summary)
    print(f"\nMain condition {MAIN}, n = {int(main.n.max())}:")
    print(main[["M", "width_med", "w90_med", "w50_med", "frac_censored", "regret_wc", "regret_inst",
                "regret_oracle", "frac_unsolved"]].round(3).to_string(index=False))
    print(f"\nVERDICT: {v}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", default="results.csv")
    r.add_argument("--quick", action="store_true")
    r.add_argument("--fine", action="store_true", help="dense lambda grid over 10^0.5..10^2.5")
    r.add_argument("--workers", type=int, default=os.cpu_count())
    r.add_argument("--seed", type=int, default=2026)
    a = sub.add_parser("analyze")
    a.add_argument("--csv", default="results.csv")
    a.add_argument("--outdir", default="report")
    a.add_argument("--alpha", type=float, default=0.9)
    a.add_argument("--min-peak", type=float, default=0.05)
    args = ap.parse_args()
    cmd_run(args) if args.cmd == "run" else cmd_analyze(args)


if __name__ == "__main__":
    main()
