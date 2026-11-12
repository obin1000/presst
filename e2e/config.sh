# Configuration for the end-to-end benchmarks. Sourced by the other scripts; override any value
# from the environment, e.g. `TPCH_SF=1 ./e2e/run-vortex.sh`.

# Where forks, data and results go (needs ~200 GB free for SF 10 plus ClickBench).
: "${E2E_WORK:=$HOME/presst-e2e}"

# Pinned forks (branch `presst` on each).
: "${VORTEX_REPO:=https://github.com/obin1000/vortex.git}"
: "${VORTEX_COMMIT:=a3073bbb0}"
: "${DUCKDB_REPO:=https://github.com/obin1000/duckdb.git}"
: "${DUCKDB_COMMIT:=8ddcc2c8}"

# Scale factors. Dictionaries for TPC-H / TPC-DS are trained on TRAIN_SF data, never on the
# benchmark data itself.
: "${TPCH_SF:=10}"
: "${TPCDS_SF:=10}"
: "${TRAIN_SF:=0.1}"

# Runs per query (the median is reported).
: "${ITERATIONS:=5}"

# Public BI workbooks with several tables: dictionaries are trained on tables 2..n and the
# queries run on table 1.
: "${PUBLIC_BI_DATASETS:=CommonGovernment Arade Bimbo CMSprovider Euro2016 Food Generico MLB Medicare1 Motos Redfin1 Telco}"

# Suites to run.
: "${SUITES:=tpch tpcds clickbench public-bi}"

PRESST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VORTEX_DIR="$E2E_WORK/vortex"
DUCKDB_DIR="$E2E_WORK/duckdb"
DICTS_DIR="$E2E_WORK/dicts"
RESULTS_DIR="$E2E_WORK/results"
VORTEX_DATA="$VORTEX_DIR/vortex-bench/data"
VORTEX_BIN="$VORTEX_DIR/target/release"
export E2E_WORK PRESST_ROOT VORTEX_DIR DUCKDB_DIR DICTS_DIR RESULTS_DIR VORTEX_DATA VORTEX_BIN

log() { printf '\n== %s ==\n' "$*" >&2; }
