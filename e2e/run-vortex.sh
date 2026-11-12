#!/usr/bin/env bash
# Vortex: default Vortex files vs. vortex-presst files (the same, with FSST tables replaced by the
# column's pretrained dictionary), queried with DataFusion and with DuckDB; Parquet as reference.
#
# Per suite it records, in $RESULTS_DIR/vortex/<suite>/:
#   ingest.tsv     format, seconds to convert the Parquet data, bytes on disk
#   datafusion.json, duckdb.json   per-query median times (gh-json lines)
set -euo pipefail
source "$(dirname "$0")/config.sh"
FORMATS="parquet,vortex,vortex-presst"

run_suite() { # name dict-dir data-gen-args...
    local name=$1 dicts=$2; shift 2
    local out="$RESULTS_DIR/vortex/$name"
    mkdir -p "$out"
    export PRESST_DICT_DIR="$dicts"
    [ -f "$PRESST_DICT_DIR/manifest.json" ] || { echo "no dictionaries in $dicts (run dicts.sh)" >&2; exit 1; }

    log "$name: base data"
    "$VORTEX_BIN/data-gen" "$@" --formats parquet
    printf 'format\tseconds\tbytes\n' >"$out/ingest.tsv"
    for format in vortex vortex-presst; do
        # Convert from scratch so the time covers the whole write.
        local dir_name=$([ "$format" = vortex ] && echo vortex-file-compressed || echo vortex-presst)
        find "$VORTEX_DATA" -type d -name "$dir_name" -path "*$DATA_PATTERN*" -prune -exec rm -rf {} +
        log "$name: writing $format"
        local start end
        start=$(date +%s.%N)
        "$VORTEX_BIN/data-gen" "$@" --formats "$format"
        end=$(date +%s.%N)
        local dir bytes
        dir=$(find "$VORTEX_DATA" -type d -name "$dir_name" -path "*$DATA_PATTERN*" | head -1)
        bytes=$(du -sb "$dir" | cut -f1)
        printf '%s\t%s\t%s\n' "$format" "$(echo "$end - $start" | bc)" "$bytes" >>"$out/ingest.tsv"
    done
    local parquet_bytes
    parquet_bytes=$(du -sb "$(find "$VORTEX_DATA" -type d -name parquet -path "*$DATA_PATTERN*" | head -1)" | cut -f1)
    printf 'parquet\t\t%s\n' "$parquet_bytes" >>"$out/ingest.tsv"

    for engine in datafusion duckdb; do
        log "$name: $engine"
        "$VORTEX_BIN/$engine-bench" "$@" --formats "$FORMATS" -i "$ITERATIONS" \
            -d gh-json -o "$out/$engine.json" --hide-progress-bar
    done
}

for suite in $SUITES; do
    case $suite in
    tpch)
        DATA_PATTERN="/tpch/$TPCH_SF/" run_suite tpch "$DICTS_DIR/tpch" tpch --opt scale-factor="$TPCH_SF" ;;
    tpcds)
        DATA_PATTERN="/tpcds/$TPCDS_SF/" run_suite tpcds "$DICTS_DIR/tpcds" tpcds --opt scale-factor="$TPCDS_SF" ;;
    clickbench)
        DATA_PATTERN="clickbench" run_suite clickbench "$DICTS_DIR/clickbench" clickbench ;;
    public-bi)
        for dataset in $PUBLIC_BI_DATASETS; do
            [ -d "$DICTS_DIR/public-bi-$dataset" ] || continue
            DATA_PATTERN="$dataset" run_suite "public-bi-$dataset" "$DICTS_DIR/public-bi-$dataset" \
                public-bi --opt dataset="$dataset"
        done
        ;;
    esac
done
log "results in $RESULTS_DIR/vortex"
