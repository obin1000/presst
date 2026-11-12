#!/usr/bin/env bash
# Clone and build the pinned Vortex and DuckDB forks, and presst's tools.
# Needs: git, a Rust toolchain (rustup), cmake, ninja, a C++17 compiler, libssl-dev (DuckDB httpfs).
set -euo pipefail
source "$(dirname "$0")/config.sh"
mkdir -p "$E2E_WORK"

checkout() { # repo dir commit
    if [ ! -d "$2/.git" ]; then
        git clone --filter=blob:none "$1" "$2"
    fi
    git -C "$2" fetch --quiet origin
    git -C "$2" checkout --quiet "$3"
    git -C "$2" log --oneline -1
}

log "presst tools"
cargo build --release --manifest-path "$PRESST_ROOT/Cargo.toml" -p xtask

log "Vortex fork"
checkout "$VORTEX_REPO" "$VORTEX_DIR" "$VORTEX_COMMIT"
(cd "$VORTEX_DIR" && cargo build --release --bin data-gen --bin datafusion-bench --bin duckdb-bench)

log "DuckDB fork"
checkout "$DUCKDB_REPO" "$DUCKDB_DIR" "$DUCKDB_COMMIT"
(cd "$DUCKDB_DIR" && GEN=ninja BUILD_BENCHMARK=1 BUILD_TPCH=1 BUILD_TPCDS=1 BUILD_HTTPFS=1 \
    CORE_EXTENSIONS="tpch;tpcds;parquet;httpfs" make release)

log "done"
echo "Vortex: $VORTEX_BIN"
echo "DuckDB: $DUCKDB_DIR/build/release/benchmark/benchmark_runner"
