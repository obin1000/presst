# End-to-end benchmarks

Does presst make a difference inside a query engine? These scripts run standard benchmarks with
and without pretrained dictionaries, in two integrations:

| Integration | What changes with presst | Engines | Suites |
|---|---|---|---|
| **Vortex** ([`obin1000/vortex`](https://github.com/obin1000/vortex), branch `presst`) | String columns that would get an FSST table per chunk use the column's pretrained table instead, stored by dictionary id (`vortex.presst`), not per chunk | DataFusion, DuckDB (reading Vortex files) | TPC-H, TPC-DS, ClickBench, Public BI |
| **DuckDB storage** ([`obin1000/duckdb`](https://github.com/obin1000/duckdb), branch `presst`, based on v2.0.0) | `DICT_FSST` builds its FSST encoder from the column's pretrained table instead of training one per segment. The table is still stored per segment, so the file format is unchanged | DuckDB | TPC-H, TPC-DS, ClickBench |

Everything else stays the same: in Vortex, OnPair and dictionary encoding still compete for every
string column exactly as in the default writer; in DuckDB, the choice between compression methods
is DuckDB's own. Queries and engines are identical; only where the FSST symbol table comes from
differs.

## Training data

Dictionaries are trained by `cargo xtask e2e-dicts` (one per string column, in FSST code order)
on data the benchmark does not query:

| Suite | Trained on | Benchmark data |
|---|---|---|
| TPC-H, TPC-DS | SF 0.1 (`TRAIN_SF`) | SF 10 |
| ClickBench | `hits_0.parquet`, 1 of 100 partitions | all 100 partitions (1% overlap) |
| Public BI | tables 2..n of a workbook | table 1, which the queries use |

TPC-H and TPC-DS text is generated from a fixed grammar, so their dictionaries transfer
unusually well: treat them as a best case. ClickBench and Public BI are real data.

## Running (Linux)

Needs Rust (rustup), git, cmake, ninja, a C++17 compiler, libssl-dev and about 200 GB of disk.

```sh
./e2e/setup.sh        # clone and build the forks at the pinned commits (~1 h)
./e2e/dicts.sh        # generate training data and train the dictionaries
./e2e/run-vortex.sh   # Vortex: DataFusion and DuckDB on parquet, vortex, vortex-presst
./e2e/run-duckdb.sh   # DuckDB storage: DICT_FSST vs. DICT_FSST with presst tables
cargo xtask e2ereport ~/presst-e2e/results -o e2e-report.md
```

Settings are in `config.sh` (work directory, scale factors, iterations, suites, Public BI
workbooks); any of them can be overridden from the environment, e.g. `SUITES=tpch TPCH_SF=1`.

For a quick check first:

```sh
SUITES=tpch TPCH_SF=1 TPCDS_SF=1 ITERATIONS=1 ./e2e/run-vortex.sh
SUITES=tpch TPCH_SF=1 ./e2e/run-duckdb.sh   # SF 1 also checks every result against the answers
```

## What is measured

- **Query time**: median of `ITERATIONS` runs per query (hot); per suite, the geometric mean of
  per-query speedups, plus totals and the best and worst query.
- **Size**: bytes on disk per format (Vortex) or database file (DuckDB).
- **Write time**: Parquet → Vortex conversion per format; for DuckDB, the first benchmark run,
  which includes generating and loading the database.

Results land in `~/presst-e2e/results/{vortex,duckdb}/<suite>/`.

## Caveats

- In DuckDB the per-segment table (~2.3 KB per 256 KB segment) is still stored; storing only the
  dictionary id needs a format change and is not done yet.
- DuckDB's dictionary loader checks the `.pdict` structure but not its xxh3 checksum.
- A column's dictionary is found by table name and column name (Vortex) or column position
  (DuckDB, `duckdb.manifest`); tables must be created with the Parquet column order.
- Public BI has no DuckDB-native suite; it runs through Vortex only.
