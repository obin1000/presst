# Column benchmark run, 2026-11-11

Raw results behind `docs/BENCHMARKS.md`, section 1.

- `main.jsonl.gz`: `presst-colbench` main run (204 columns, all training levels).
- `stress.jsonl.gz`: `presst-colbench --stress aging,sweep,matrix,mix`.
- `dicts.tar.gz`: every presst dictionary trained in the main run
  (`<dataset>-<table>-<column>-<level>.pdict`), so results can be checked
  without retraining: `presst inspect`, `presst eval`.

Machine: Intel Core i7-7740X, Windows, rustc 1.99.1, commit of this folder.
Regenerate the tables with `cargo xtask colreport <(gunzip -c main.jsonl.gz)`
or after unpacking.
