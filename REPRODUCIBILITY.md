# Reproducibility checklist

The artifact contains the frozen QAP-FL checkpoint, all data required by the
three main tables, the exact variable-budget definitions, and the two BLS
backends used by the runner. The command-line entry point is
`run_main_tables.py`.

For a quick validation, run one instance from each dataset with `--limit 1`.
For the reported tables, omit `--limit`; the default synthetic sample count
is 256 per distribution and size. Output CSV files record the instance, cost,
gap, wall-clock time, candidate count, and number of starts.

The repository intentionally does not contain author names, affiliations,
absolute workstation paths, or generated result logs. External baseline
implementations are not copied into this minimal artifact.
