#!/usr/bin/env bash
# DuckDB native storage: DICT_FSST as shipped (an FSST table trained per segment) vs. DICT_FSST
# with each column's pretrained presst table (PRESST_DICT_DIR set). Same binary, same queries;
# DuckDB's benchmark runner checks every result against the reference answers where it has them
# (TPC-H and TPC-DS answers exist for SF 1, so run with TPCH_SF=1 TPCDS_SF=1 once to check results).
#
# Per suite it records, in $RESULTS_DIR/duckdb/<suite>/:
#   <variant>.tsv        benchmark_runner output (name, run, seconds)
#   <variant>.load.txt   seconds to build the database, database size, storage info of VARCHAR columns
set -euo pipefail
source "$(dirname "$0")/config.sh"
RUNNER="$DUCKDB_DIR/build/release/benchmark/benchmark_runner"
DUCKDB_CLI="$DUCKDB_DIR/build/release/duckdb"
cd "$DUCKDB_DIR"

# ClickBench: load the partitions data-gen already downloaded instead of fetching 14 GB per variant.
use_local_clickbench() {
    local dir
    dir=$(find "$VORTEX_DATA" -type d -name parquet -path "*clickbench*" | head -1)
    git checkout -- benchmark/clickbench/queries/load.sql
    sed -i "s|https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_{}.parquet|$dir/hits_{}.parquet|" \
        benchmark/clickbench/queries/load.sql
}

run_suite() { # name regex first-benchmark runner-args...
    local name=$1 pattern=$2 first=$3; shift 3
    local out="$RESULTS_DIR/duckdb/$name"
    mkdir -p "$out"
    for variant in dict_fsst presst; do
        log "$name: $variant"
        # Cached databases must not be shared between variants.
        rm -rf duckdb_benchmark_data
        if [ "$variant" = presst ]; then
            export PRESST_DICT_DIR="$DICTS_DIR/$name"
        else
            unset PRESST_DICT_DIR
        fi
        local start end
        start=$(date +%s.%N)
        # The first benchmark builds and caches the database; time it separately.
        "$RUNNER" "$first" --threads="$(nproc)" --disable-timeout "$@" --out="$out/$variant.first.tsv"
        end=$(date +%s.%N)
        echo "first pass (includes loading) seconds: $(echo "$end - $start" | bc)" >"$out/$variant.load.txt"
        local db
        db=$(find duckdb_benchmark_data -name '*.db' -o -name '*.duckdb' | head -1)
        if [ -n "$db" ]; then
            echo "database bytes: $(du -sb "$db" | cut -f1)" >>"$out/$variant.load.txt"
            # pragma_storage_info only takes a literal table name: one query per table.
            local tables query=""
            tables=$("$DUCKDB_CLI" -readonly -noheader -list "$db" -c "SELECT table_name FROM duckdb_tables() ORDER BY 1;")
            for t in $tables; do
                query+="SELECT '$t' AS table_name, column_name, compression, count(*) AS segments
                        FROM pragma_storage_info('$t') WHERE segment_type = 'VARCHAR' GROUP BY ALL UNION ALL "
            done
            [ -n "$query" ] && "$DUCKDB_CLI" -readonly -csv "$db" -c "${query% UNION ALL } ORDER BY ALL;" \
                >>"$out/$variant.load.txt" || true
        fi
        "$RUNNER" "$pattern" --threads="$(nproc)" "$@" --out="$out/$variant.tsv"
    done
}

for suite in $SUITES; do
    case $suite in
    tpch) run_suite tpch "benchmark/tpch/sf1/.*" benchmark/tpch/sf1/q01.benchmark --sf "$TPCH_SF" ;;
    tpcds) run_suite tpcds "benchmark/tpcds/sf1/.*" benchmark/tpcds/sf1/q01.benchmark --sf "$TPCDS_SF" ;;
    clickbench)
        use_local_clickbench
        run_suite clickbench "benchmark/clickbench/.*" benchmark/clickbench/q00.benchmark
        git checkout -- benchmark/clickbench/queries/load.sql
        ;;
    public-bi) echo "public-bi: DuckDB has no Public BI suite; run-vortex.sh covers it" >&2 ;;
    esac
done
log "results in $RESULTS_DIR/duckdb"
