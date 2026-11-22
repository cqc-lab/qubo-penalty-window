# Provenance of the QPU data

All QPU runs were submitted from one Windows 10 workstation (Python 3.10.5, `requirements-qpu.lock`) to D-Wave
Leap. The emulation runs were made on a 32-core Linux server (Python 3.12.3, `requirements-emulation.lock`).
Dates are local time (KST, UTC+9). QPU times are QPU access time as reported by the solver; the per-run sums
below add up exactly to the ledger total (`data/qpu/qpu_ledger.json`, 925.2 s; rows are rounded).

| File | Run (paper, Table 3) | Solver | Date | QPU time (s) | Script and options |
|---|---|---|---|---|---|
| `qpu_pilot.csv` | pilot | Advantage_system4 | 2026-10-27 | 12.8 | `qpu_run.py --phase pilot` |
| `qpu_main.csv` | main | Advantage_system4 | 2026-10-27 | 82.7 | `qpu_run.py --phase main` |
| `qpu_estimator.csv` | selection | Advantage_system4 | 2026-10-27 | 61.1 | `qpu_estimator.py` |
| `qpu_chain.csv` | chain | Advantage_system4 | 2026-10-27 | 34.4 | `qpu_run.py --phase chain` |
| `qpu_bign.csv` | large-n | Advantage_system4 | 2026-10-27 | 63.5 | `qpu_run.py --phase bign` |
| `qpu_garcia.csv` | published methods | Advantage_system6 | 2026-10-28 (44 instances), 2026-10-29 (last 4, about ten hours later) | 49.6 | `qpu_garcia.py --solver Advantage_system6` |
| `qpu_c1.csv` | λ/λ_exact sweep | Advantage_system4 | 2026-10-30 | 82.2 | `qpu_c1.py --ns 10 18` |
| `qpu_c2.csv` | held-out | Advantage_system4 | 2026-10-30 | 96.1 | `qpu_c2.py` |
| `qpu_interleave.csv` | interleaved | Advantage_system6 | 2026-11-05 | 68.0 | `qpu_interleave.py --solver Advantage_system6` |
| `qpu_large.csv` (sparse) | large sparse | Advantage_system6 | 2026-11-12/13 | 109.4 | `qpu_large.py --family sparse --solver Advantage_system6` |
| `qpu_large.csv` (dense) | large dense | Advantage_system6 | 2026-11-13 | 7.6 | `qpu_large.py --family dense --solver Advantage_system6` |
| `qpu_apps.csv` | applications | Advantage_system6 | 2026-11-13 | 19.5 | `qpu_apps.py --solver Advantage_system6` |
| `qpu_strength.csv` | strength | Advantage_system6 | 2026-11-17 | 25.3 | `qpu_strength.py --budget 60 --solver Advantage_system6` |
| `qpu_budget.csv` | equal budget | Advantage_system6 | 2026-11-17 | 39.3 | `qpu_budget.py --budget 60 --solver Advantage_system6` |
| `qpu_adv2_main.csv` | Advantage2 | Advantage2_system1 | 2026-11-17 | 97.4 | `qpu_run.py --phase main --budget 130 --solver Advantage2_system1 --emb-file qpu_adv2_embeddings.json --out qpu_adv2_main.csv` |
| `qpu_budget_adv2.csv` | equal budget, Advantage2 | Advantage2_system1 | 2026-11-17 | 76.3 | `qpu_budget.py --budget 110 --solver Advantage2_system1 --methods est,seq15,inst --out qpu_budget_adv2.csv` |

Notes
- The first five runs were started without `--solver`; the Leap default for Pegasus at the time was
  Advantage_system4, as recorded in the run logs. Only `qpu_c1.csv` and `qpu_c2.csv` also record the chip in a
  column. Pass `--solver` explicitly when re-running.
- The published-methods run was served by Advantage_system6. It used the same clique embeddings
  (`qpu_embeddings.json`, computed on Advantage_system4), which were valid on both systems. So did the interleaved chain-length run (`qpu_interleave.csv`), made after
  Advantage_system4 had been retired; it records the chip in a column, as do the large-instance runs (`qpu_large.csv`).
- The sparse large-instance run stopped once at instance 8 of the first cell, because an embedding computed
  from the coefficient matrix at λ = 1 missed couplings that cancel at that λ only. The script was fixed (all
  nonzero couplings at any λ) and resumed. The error is raised before a problem is submitted, so the aborted
  instance used no QPU time, and no partial instance was saved.
- The QPU run of `qpu_garcia.csv` stopped after 44 of 48 instances because the project's solver time ran out;
  the remaining four instances were run with a different Leap token, on the same system.
- `qpu_main.csv` and `qpu_pilot.csv` were written by an earlier version of `qpu_run.py` without the `phase` and
  `chain_len` columns. These two columns were added afterwards from the run settings and
  `qpu_embeddings.json`, so that all sweep files share the current schema; no measured value was changed.
- `qpu_estimator.csv`, `qpu_garcia.csv` and `qpu_c2.csv` store per-instance QPU time in `select_us` and
  `eval_us`, repeated on every method row of an instance; the other files store it per row in `qpu_us`.
- Chain strength was Ocean's default (uniform torque compensation) in every run, except the `scaled` setting
  of `qpu_c1.csv`, where it equals the logical auto-scale factor s(λ).
- Every run left `auto_scale` at its default (on) and the annealing time at its default (20 µs). Since
  2023-11-15, Leap's auto-scaling uses the extended J range [-2, 1] and, on Advantage systems, the per-qubit
  coupling limit [-18, 15]; all runs in this study were made after that date. Every programming also carried a
  random spin-reversal gauge (two per point), which is not stored in the CSV files but is reproducible from the
  scripts' seeds; `embedded_gauge.py` reconstructs the gauges and the auto-scale factor of every programming.
- The four runs of 2026-11-17 were submitted with a different Leap token (a new project), from the same workstation.
  `qpu_strength.py` programs the embedded problem itself with `auto_scale=False`, after dividing it by the factor
  that auto-scaling would apply (extended J range, per-qubit coupling limit); its gauges are stored implicitly
  through the seeds (`default_rng(instance seed + 53)`).
- The protocol of the Advantage2 equal-budget run (methods, margin 1.5, 1500 reads per method including selection,
  criterion within 1% of E_ref) was fixed in the docstring of `qpu_budget.py` before the run. Its embeddings were
  computed with `embed_large.py` on the Advantage2 working graph (not redistributed) and are stored in
  `qpu_large_embeddings.json` under the prefix `Advantage2_system1:`.
