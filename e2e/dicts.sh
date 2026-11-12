#!/usr/bin/env bash
# Train one presst dictionary per string column for each suite, on data the benchmark never
# queries (or, for ClickBench, 1 of its 100 partitions):
#
#   tpch, tpcds   TRAIN_SF data (default 0.1); the benchmark runs at TPCH_SF / TPCDS_SF
#   clickbench    hits_0.parquet, 1 of the 100 partitions the benchmark loads (~1% overlap)
#   public-bi     tables 2..n of each workbook; the queries run on table 1
#
# Output: $DICTS_DIR/<suite>/{manifest.json,duckdb.manifest,<id>.pdict}. Each suite gets its own
# directory because table names repeat across suites (TPC-H and TPC-DS both have `customer`).
set -euo pipefail
source "$(dirname "$0")/config.sh"
XTASK="$PRESST_ROOT/target/release/xtask"
mkdir -p "$DICTS_DIR"

# The Parquet directory data-gen wrote for a benchmark (first match of a path pattern).
parquet_dir() {
    local dir
    dir=$(find "$VORTEX_DATA" -type d -name parquet -path "$1" | sort | head -1)
    [ -n "$dir" ] || { echo "no Parquet directory matching $1 under $VORTEX_DATA" >&2; exit 1; }
    echo "$dir"
}

train() { # name inputs...
    local out="$DICTS_DIR/$1"; shift
    rm -rf "$out"
    "$XTASK" e2e-dicts -o "$out" "$@"
}

for suite in $SUITES; do
    case $suite in
    tpch | tpcds)
        log "$suite: generating SF $TRAIN_SF training data"
        "$VORTEX_BIN/data-gen" "$suite" --formats parquet --opt scale-factor="$TRAIN_SF"
        train "$suite" "$(parquet_dir "*/$suite/$TRAIN_SF/*")"
        ;;
    clickbench)
        log "clickbench: downloading the benchmark data, training on hits_0"
        "$VORTEX_BIN/data-gen" clickbench --formats parquet
        train clickbench --table hits "$(parquet_dir "*clickbench*")/hits_0.parquet"
        ;;
    public-bi)
        for dataset in $PUBLIC_BI_DATASETS; do
            log "public-bi $dataset: training on tables 2..n"
            "$VORTEX_BIN/data-gen" public-bi --formats parquet --opt dataset="$dataset"
            dir=$(parquet_dir "*$dataset*")
            mapfile -t others < <(find "$dir" -name '*.parquet' ! -name '*_1.parquet' | sort)
            if [ "${#others[@]}" -eq 0 ]; then
                echo "  $dataset has a single table: no dictionaries (default compression)" >&2
                continue
            fi
            train "public-bi-$dataset" "${others[@]}"
        done
        ;;
    *) echo "unknown suite $suite" >&2; exit 1 ;;
    esac
done
log "dictionaries in $DICTS_DIR"
ls "$DICTS_DIR"
