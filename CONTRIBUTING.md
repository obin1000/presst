# Contributing

## Checks

CI runs these on every pull request; run them locally first:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

The core crate must build without `std`:

```sh
cargo build -p presst --no-default-features --target thumbv7em-none-eabihf
```

## Unsafe code

The hot loops in `crates/presst/src/codec/` use `unsafe` for unaligned loads
and stores. Every `unsafe` block needs a `// SAFETY:` comment (enforced by
clippy), and every fast kernel must stay byte-identical to the slow oracle in
`crates/presst/src/reference.rs`. The property tests in
`crates/presst/tests/properties.rs` check this; Miri runs nightly:

```sh
cargo +nightly miri test -p presst
```

## Fuzzing

Fuzz targets live in `fuzz/` (Linux or macOS; libFuzzer is unreliable on
Windows MSVC):

```sh
cargo +nightly fuzz run decode_frame -- -max_total_time=600
```

Targets: `decode_frame`, `decode_block`, `parse_dict`, `roundtrip`, `strings`.

## Binary formats

`docs/FORMAT.md` is the normative spec. The golden files in
`crates/presst/tests/golden/` catch accidental format changes. If a change is
intended, bump the format version and regenerate them:

```sh
PRESST_BLESS=1 cargo test -p presst --test golden
```

## Benchmarks

```sh
cargo xtask corpus                       # download and verify the dbtext corpus
cargo run --release -p presst-bench --bin presst-eval -- corpora/data/dbtext/* --json results.jsonl
cargo xtask report results.jsonl         # Markdown summary
cargo bench -p presst-bench              # microbenchmarks
```

`presst-eval` trains on the first half of each file and measures on the
second half, so dictionaries are never evaluated on their training data.

The main benchmark runs on real string columns (about 3.5 GB of downloads,
pinned by checksum in `corpora/*.sources`; see `docs/BENCHMARKS.md`):

```sh
cargo xtask data all
cargo run --release -p presst-bench --bin presst-colbench -- --quick           # smoke run
cargo run --release -p presst-bench --bin presst-colbench -- --json main.jsonl # full run, ~1.5 h
cargo xtask colreport main.jsonl --svg chart.svg
```
