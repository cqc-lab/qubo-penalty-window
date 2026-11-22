# The penalty window collapses on quantum annealing hardware

Code and data for *The Penalty Window Collapses on Quantum Annealing Hardware: A Scaling Variable and a
Sample-Based Rule for Penalty Weights in Constrained QUBO* (J. Cho, M. Lee, I. Kim, D. Lee).

Repository: https://github.com/cqc-lab/qubo-penalty-window  
Archive: [doi:10.5281/zenodo.24281103](https://doi.org/10.5281/zenodo.24281103) (all versions; v1.0.4 is
[doi:10.5281/zenodo.24439026](https://doi.org/10.5281/zenodo.24439026))

The study asks how the penalty weight λ of a penalty-encoded constrained QUBO should be chosen when the problem
runs on a D-Wave QPU. It contains
- sweeps of λ in software simulated annealing (SA), in an SA emulation of QPU conditions (auto-scaling to the
  hardware coefficient ranges, fixed temperature, control noise), and on D-Wave Advantage systems;
- a scaling-variable analysis of the upper edge of the penalty window and fits of an effective inverse temperature;
- a sample-based selection rule (`est`) compared with sequential, grid, coefficient-based and published methods,
  in emulation (including held-out instance families and n = 50–100) and on hardware.

## Layout

```
code/                       all scripts
data/emulation/             SA and emulation results
data/qpu/                   raw D-Wave results, the QPU-time ledger and the clique embeddings used
data/derived/               small summaries produced by the analysis scripts
PROVENANCE.md               solver, date, QPU time and command of every QPU run
requirements.txt            minimum versions
requirements-*.lock         exact versions of the emulation (Linux) and QPU (Windows) environments
```

All instances are generated from fixed seeds inside the code, so no instance files are stored. Instances with
n ≤ 22 are solved exactly by enumeration.

## Setup

```bash
python -m venv venv
. venv/bin/activate                 # Windows: venv\Scripts\activate
pip install -r requirements.txt     # or one of the lock files for the exact versions used
```

The scripts read and write files in the current directory. To **analyse the published data**, copy code and
data into one working directory:

```bash
mkdir work && cp code/*.py data/emulation/* data/qpu/* data/derived/* work/ && cd work
python scaling.py --csv results.csv --outdir report     # writes report/scaling_points.csv, needed below
```

To **re-run an experiment from scratch**, use a working directory with the code only (plus
`qpu_embeddings.json` if you want the same embeddings). The scripts skip work already present in their output
file and reuse cached grids (`qpu_beta_grid*.csv`, `beta_fine_grid.csv`, `embedded_scale.csv`) when these exist,
so copying the published results would make a re-run do nothing. Do not copy `qpu_ledger.json` either: the QPU
scripts refuse to run once the ledger reaches the monthly cap (`--month-cap`, default 560 s).

The QPU scripts need a D-Wave Leap account with an Ocean token configured (`dwave config create`).
**Never commit a token**; `.gitignore` excludes the usual file names. Every QPU script takes `--budget`
(seconds of QPU access time for one invocation), `--solver`, and `--dry-run` (SA on a Pegasus-shaped graph,
no QPU time), and logs its usage to `qpu_ledger.json`. Pass `--solver` explicitly: the solvers used are listed
in `PROVENANCE.md`.

## Experiments, scripts and data

Commands assume the analysis working directory above. Table and figure numbers refer to the submitted
manuscript and may shift in revision.

| Paper | Command | Data |
|---|---|---|
| Coarse λ sweep, conditions C0–C4, S1, S2 (Fig. 1) | `gonogo.py run --out results.csv`; `gonogo.py analyze --csv results.csv --outdir report` | `results.csv`, `per_instance_coarse.csv`, `summary_coarse.csv` |
| Fine λ sweep, window widths (Table A1) | `gonogo.py run --fine --out results_fine.csv`; `gonogo.py analyze --csv results_fine.csv --outdir report_fine` | `results_fine.csv`, `per_instance_fine.csv`, `summary_fine.csv` |
| Scaling-variable collapse in emulation (Fig. 2, Table 4) | `scaling.py --csv results.csv --outdir report` | `report/scaling_points.csv` (generated) |
| QPU sweeps (Fig. 1, Tables 3, 4) | `qpu_run.py --phase {pilot,main,chain,bign} --solver Advantage_system4` | `qpu_pilot.csv`, `qpu_main.csv`, `qpu_chain.csv`, `qpu_bign.csv` |
| QPU vs. emulation, paired | `qpu_compare.py qpu_main.csv --out qpu_main_compare.csv`; `qpu_compare.py qpu_pilot.csv --out qpu_pilot_compare.csv` | `qpu_main_compare.csv`, `qpu_pilot_compare.csv` |
| Effective temperature, coarse fits (Fig. 3) | `qpu_beta.py --qpu qpu_main.csv --sa results.csv --out qpu_beta_grid.csv > qpu_beta_fit.txt`; the same with `--qpu qpu_chain.csv --out qpu_beta_grid_chain.csv` and `--qpu qpu_bign.csv --out qpu_beta_grid_bign.csv` | `qpu_beta_grid*.csv`, `qpu_beta_fit*.txt` |
| Effective temperature, fine direct fit with intervals (Fig. 4, Table 5) | `beta_fine.py` | `beta_fine_grid.csv`, `beta_fine.csv` |
| Embedded-problem scale, scaling fit with intervals (symmetric ranges, superseded) | `embedded_scale.py` | `embedded_scale.csv`, `embedded_beta.csv` |
| Hardware scale per programming with the reconstructed spin-reversal gauges, chain-strength comparison, scaling fit (Table 5, Section 6.3) | `embedded_gauge.py --workers 30` | `embedded_gauge.csv`, `embedded_gauge_beta.csv` |
| Paired gauge test of the temperature scale (Section 6.3) | `gauge_pairs.py --beta 2.9` (reads `embedded_gauge.csv`) | `gauge_pairs.csv` |
| Hardware scale without gauges (superseded) | `embedded_autoscale.py --workers 30` | `embedded_autoscale.csv`, `embedded_autoscale_beta.csv` |
| TSP and knapsack encodings, Γ (Table B1) | `apps.py sweep --out apps_sweep.csv`; `apps.py est --out apps_est.csv` | `apps_sweep.csv`, `apps_est.csv` |
| TSP and knapsack on the QPU, Γ estimated from samples | `qpu_apps.py --solver Advantage_system6` | `qpu_apps.csv`, `qpu_apps_embeddings.json` |
| Controlled test of the programmed energy scale (Section 6.3) | `qpu_strength.py --solver Advantage_system6`; `strength_analyze.py --csv qpu_strength.csv` | `qpu_strength.csv`, `strength_summary.csv` |
| Equal-budget comparison on the large sparse instances (Section 7.4) | `qpu_budget.py --solver Advantage_system6` | `qpu_budget.csv` |
| Equal-budget comparison on Advantage2, with seq x 1.5 (Section 7.4) | `embed_large.py --graph graph_Advantage2_system1.json --chip Advantage2_system1`; `qpu_budget.py --budget 110 --solver Advantage2_system1 --methods est,seq15,inst --out qpu_budget_adv2.csv` | `qpu_budget_adv2.csv`, `qpu_large_embeddings.json` |
| Main sweep on Advantage2 (Section 6.3) | `qpu_run.py --phase main --solver Advantage2_system1 --emb-file qpu_adv2_embeddings.json --out qpu_adv2_main.csv`; `beta_adv2.py --workers 30` | `qpu_adv2_main.csv`, `qpu_adv2_embeddings.json`, `beta_adv2_grid.csv`, `beta_adv2.csv` |
| Out-of-sample R² of the scaling variable and control variables | `oos_scaling.py --qpu qpu_main.csv --points report/scaling_points.csv` | `oos_scaling.csv` |
| Multiplicity (Holm) over all paired hardware comparisons | `holm_qpu.py --dir . --out holm_qpu.csv` | `holm_qpu.csv` |
| Selection rules in emulation, published methods (Tables C1, C2) | `estimator.py --out estimator.csv` | `estimator.csv` |
| Margin sensitivity, per-instance best, cross-entropy, held-out families (Table D1) | `extra.py --family {orig,rho075,gauss,sparse,onehot}` | `extra_*.csv` |
| Large instances, n = 50–100 (Table E1) | `large.py --out large.csv`; `tabu_ref.py --ref large.csv` | `large.csv`, `tabu_ref.csv` |
| Large instances at hardware-calibrated β (Table 7) | `large_beta.py --ref large.csv --out large_beta.csv` | `large_beta.csv` |
| Large instances on the QPU (Table 8) | `qpu_large.py --family sparse --solver Advantage_system6`; `qpu_large.py --family dense --solver Advantage_system6`; `qpu_large_analyze.py qpu_large.csv --large large.csv --beta large_beta.csv` | `qpu_large.csv`, `qpu_large_embeddings.json`, `qpu_large_summary.csv` |
| Held-out one-hot family on the QPU (Table F1) | `qpu_c2.py --solver Advantage_system4` | `qpu_c2.csv` |
| Published methods on the QPU (Table F2) | `qpu_garcia.py --solver Advantage_system6` | `qpu_garcia.csv` |
| Selection rules on the QPU (Table 6) | `qpu_estimator.py --solver Advantage_system4` | `qpu_estimator.csv` |
| Chain length vs. β_eff, both embeddings interleaved in one session | `qpu_interleave.py --solver Advantage_system6`; `interleave_analyze.py qpu_interleave.csv --grid beta_fine_grid.csv` | `qpu_interleave.csv`, `interleave_beta.csv` |
| λ/λ_exact sweep and chain-strength test on the QPU | `qpu_c1.py --ns 10 18 --solver Advantage_system4`; `c1_analyze.py qpu_c1.csv` | `qpu_c1.csv`, `c1_beta.csv` |
| Time to solution | `tts.py estimator.csv qpu_estimator.csv qpu_garcia.csv qpu_c2.csv` | — |
| Figures 1 and 4 | `figs.py figures`; `fig_beta.py beta_fine.csv embedded_beta.csv figures/beta_chain.png interleave_beta.csv qpu_large_summary.csv` | — |

Figures 2 and 3 are written by `scaling.py` (to `report/`) and `qpu_beta.py` (to the working directory).

## Dependencies between steps

- `embedded_scale.py` and `c1_analyze.py` read `report/scaling_points.csv` (from `scaling.py`).
- `embedded_gauge.py`, `embedded_autoscale.py`, `c1_analyze.py` and `embedded_scale.py` need the coupler graph of the device.
  The full working graph of the device is not redistributed. These analyses only need the couplers among the
  qubits of each embedding, which `data/qpu/embedding_couplers_Advantage_system4.json` contains (1060 couplers,
  made by `used_couplers.py` from the graph fetched with `fetch_graph.py` while Advantage_system4 was online;
  results are identical to those with the full graph). Copy it into the working directory; the scripts use
  `graph_Advantage_system4.json` instead if that file is present. The ranges used by auto-scaling (`h_range`,
  `j_range`, `extended_j_range`, `per_qubit_coupling_range`) and the default annealing time are in
  `data/qpu/qpu_solver_properties.json` (read on 2026-11-16 from Advantage_system6; Advantage_system4 was retired
  by then and is assumed to have had the same ranges).
- `embedded_gauge.py` is the analysis the paper reports (Table 5, Section 6.3). The QPU scripts applied a random
  spin-reversal gauge to every programming, and under the extended J range [-2, 1] and the signed per-qubit
  coupling limit the auto-scale factor depends on that gauge. The script reconstructs the 4800 gauges of the main,
  chain, large-n and chain-strength runs from the scripts' random streams (`default_rng(seed + 7)` with the shuffled
  job list in `qpu_run.py`; `default_rng(instance seed + 41)` per instance in `qpu_c1.py`), checks the
  reconstructed job order against the CSV row order, and computes the factor per programming.
  `embedded_autoscale.py` (actual ranges, no gauge) and `embedded_scale.py` (symmetric ranges, no gauge) are
  earlier versions, kept for comparison.
- `qpu_garcia.py` reads the λ chosen by `est` from `qpu_estimator.csv` (`--est`).
- `qpu_beta.py`, `beta_fine.py` and `embedded_scale.py` emulate the QPU instances once and cache the result in
  their grid file; delete it to recompute.

## Notes

- **Hardware.** All QPU runs used D-Wave Advantage_system4 except the published-methods comparison
  (`qpu_garcia.csv`), the interleaved chain-length run (`qpu_interleave.csv`), the large instances (`qpu_large.csv`) and
  the application encodings (`qpu_apps.csv`), the coupling-strength run (`qpu_strength.csv`) and the equal-budget run
  (`qpu_budget.csv`), which ran on Advantage_system6, and the repeat of the main sweep on Advantage2_system1
  (`qpu_adv2_main.csv`) and the second equal-budget run (`qpu_budget_adv2.csv`). Total QPU access time was 925.2 s
  (`qpu_ledger.json`).
  See `PROVENANCE.md`.
- **Embeddings.** `qpu_embeddings.json` holds the clique embeddings (`minorminer.busclique`) used on the QPU,
  keyed by problem size; `"10of14"` and `"18of30"` are the lengthened chains of the chain-length experiment.
  `qpu_large_embeddings.json` holds the minorminer embeddings of the large sparse instances and the clique
  embedding of K50 on Advantage_system6, keyed by solver and instance.
- **Coarse and fine β fits.** `qpu_beta_fit*.txt` are the coarse-grid fits (β step 0.5). The paper's Table 5 and
  Fig. 4 use the finer fits in `beta_fine.csv` and `embedded_beta.csv`; numbers differ slightly between them
  (for example 3.9 vs. 3.65 for the scaling fit of the lengthened-chain n = 10 run).
- **Emulation scripts** use multiprocessing (`--workers`); the full emulation was run on a 32-core server.
- **SA sampler.** The code uses `dwave-samplers`; `gonogo.py` falls back to the older `neal` package only if
  `dwave-samplers` is missing.
- **Randomness.** Instance seeds derive from base seed 2026 (4242 for the held-out families in `extra.py`).
  QPU draws cannot be replayed, so QPU results reproduce only statistically.

## Methods reimplemented from other work

- Sequential penalty methods (Algorithms 1 and 3) of M. Diez García, M. Ayodele and A. Moraglio, "Exact and
  Sequential Penalty Weights in Quadratic Unconstrained Binary Optimisation with a Digital Annealer",
  GECCO '22 Companion, pp. 184–187, doi:10.1145/3520304.3528925 — `estimator.garcia_standard`, `estimator.garcia_binary`, `qpu_garcia.py`.
- Cross-entropy optimisation of penalty values of C. Roch, A. Impertro and C. Linnhoff-Popien, "Cross Entropy
  Optimization of Constrained Problem Hamiltonians for Quantum Annealing", ICCS 2021,
  doi:10.1007/978-3-030-77980-1_5 — `extra.cross_entropy`, adapted to a single penalty weight.

These are our implementations from the published descriptions, not the authors' code.

## Acknowledgements

QPU access was provided through D-Wave Leap.

## License

Code (`code/`): MIT, see `LICENSE`. Data (`data/`): CC BY 4.0, see `data/LICENSE`.

## Citation

See `CITATION.cff`. Please cite the paper; citation details will be added on publication.
