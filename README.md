# QAP-FL Anonymous Reproducibility Artifact

This bundle contains the smallest runnable path for the three QAP-FL main
tables: synthetic QAP, Taixxeyy, and QAPLIB. It includes the frozen QAP-FL
checkpoint, the 39-dimensional node feature construction, the 8-dimensional
FAQ/structural compatibility features, the 80-candidate QAP-FL portfolio, the
official C++ BLS executable used by the paper protocol, and the required data.

The runner compares the paper method with the matched single-start BLS
baseline. SAWT, PLMA, and NGM are external baselines and are intentionally not
vendored into this artifact; their reported numbers are not silently
reimplemented here. No author, institution, or local-machine identifier is
needed to run the artifact.

## Environment

Python 3.10 or 3.11 is recommended. Install the dependencies with:

```powershell
python -m pip install -r requirements.txt
```

The bundled executable is Windows x86-64. On Linux or macOS, use
`--backend python` for a portable run, or replace the executable with a
platform-compatible build of the same BLS backend.

## Smoke tests

Run one instance from each main table:

```powershell
python run_main_tables.py --dataset synthetic --samples 1 --limit 1 --backend python --output runs/synthetic_smoke.csv
python run_main_tables.py --dataset taix --limit 1 --backend python --output runs/taix_smoke.csv
python run_main_tables.py --dataset qaplib --limit 1 --backend python --output runs/qaplib_smoke.csv
```

For the paper backend on Windows, replace `--backend python` with
`--backend cpp`. The default `auto` mode selects C++ when the bundled binary is
available.

## Full main tables

The paper protocol uses 256 instances per synthetic distribution and size,
20 Taixxeyy instances per size, and all bundled QAPLIB `.dat` files:

```powershell
python run_main_tables.py --dataset synthetic --backend cpp --output runs/synthetic_main.csv
python run_main_tables.py --dataset taix --backend cpp --output runs/taix_main.csv
python run_main_tables.py --dataset qaplib --backend cpp --output runs/qaplib_main.csv
```

The output is one row per method and instance with the budget, final cost,
gap, runtime, number of starts, and candidate count. Synthetic gaps use the
frozen Ro-TS reference files in `data/synthetic/references`; Taixxeyy gaps use
the published best-known values; QAPLIB gaps use the bundled `.sln` values.

## Protocol notes

* Synthetic budgets are 2/7/40 seconds for n=20/50/100.
* Taixxeyy budgets are 2/5/20/30/40 seconds for n=27/45/75/125/175.
* QAPLIB budgets are 4 seconds for ordinary n<=100, 10 seconds for hard
  families or n>100, and 20/40 seconds for the larger Tai tiers.
* QAP-FL builds four relabeling orbits, decodes 40 learned plus 40 random
  candidates, keeps four diverse candidates per source, probes all eight, then
  refines the best learned and random survivors.
* All timing includes candidate construction, process launch, and backend
  calls. The Python backend is a portability implementation and should not be
  mixed with the paper's C++ timing results.

## Anonymous artifact submission

This directory is intended to be uploaded as a GitHub repository without
author-identifying metadata. After creating the repository, paste its URL into
<https://anonymous.4open.science/> and use the generated anonymous URL in the
paper. Generated `runs/` outputs are excluded from version control by the
provided `.gitignore`.
