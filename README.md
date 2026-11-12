# presst

**PREtrained Static Symbol Table compression**: FSST-style string compression
with symbol tables that are trained offline, once, instead of per block.

> Status: early development. The core codec, trainer, CLI and benchmark
> harness work and are tested; the binary format is not stable before 1.0.

## The idea

[FSST](https://github.com/cwida/fsst) compresses strings by replacing frequent
substrings (up to 8 bytes) with one-byte codes from a symbol table. Decoding is
one table lookup per code, so it is extremely fast, and strings stay randomly
accessible. FSST trains a fresh table for every block and stores it in the
block header.

While building a GPU implementation of FSST
([High Throughput GPU-Accelerated FSST String Compression](https://github.com/obin1000/fsst-gpu), ADMS 2025),
we saw that the tables trained for different blocks of the same kind of data
converge to very similar entries. Making blocks bigger doesn't help: it
increases the number of misses.

presst separates the two steps:

1. **Train offline.** A dictionary is trained once on representative data,
   with far more compute than per-block training can afford: large samples,
   many generations, local search.
2. **Reference by id.** Compressed data stores only the dictionary's content
   hash. There is no per-block training at compression time, no table to store
   in each block, and no table to parse at decompression time. Encode and
   decode tables are built once and stay hot.

The goal is speed, not a better compression ratio: the biggest wins are
expected for small inputs and per-string (columnar) compression, where FSST's
per-block training and table overhead dominate.

## Results

Held-out data from the 23 datasets of the FSST paper's dbtext corpus: every
dictionary or table is trained on the first half of each dataset and measured
on the second half. Single thread, Intel i7-7740X. Full method and all
scenarios in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

| | Individual strings | | 128 B blocks | | 1 MiB blocks | |
|---|---:|---:|---:|---:|---:|---:|
| **Codec** | **Ratio** | **Decompress** | **Ratio** | **Decompress** | **Ratio** | **Decompress** |
| **presst** | **2.29** | 1.3 GB/s | **2.26** | 2.6 GB/s | 2.31 | **4.3 GB/s** |
| FSST, table per block | – | – | 0.60 | 0.2 GB/s | 2.09 | 3.5 GB/s |
| FSST, table trained once | 2.07 | 1.1 GB/s | 2.06 | 2.3 GB/s | 2.10 | 3.5 GB/s |
| LZ4 | 0.95 | 1.5 GB/s | 1.07 | 3.0 GB/s | 1.61 | 2.7 GB/s |
| zstd -3 + dictionary (≤110 KiB) | 0.90 | 0.2 GB/s | 1.81 | 0.3 GB/s | 2.72 | 0.9 GB/s |

presst compresses at about 400 MB/s in all scenarios (per-block FSST: under
1 MB/s for 128 B blocks, 180 MB/s for 1 MiB blocks, because it trains a table
for every block). zstd reaches better ratios on large blocks, at a quarter of
the decompression speed. Decoding one random string takes about 40 ns.

What this shows so far:

- **Pretraining removes FSST's small-input penalty.** A per-block table costs
  more than a 128-byte block saves; a shared dictionary costs nothing per block.
- **More training compute pays off.** On unseen data, the extended trainer's
  dictionaries beat FSST's own training algorithm by about 6% in ratio (2.31 vs
  2.17 with the same presst codec), and FSST's per-block tables by about 10%
  (2.31 vs 2.09 on 1 MiB blocks).
- **The decoder is faster than fsst-rs** on large blocks, mainly because a full
  dictionary (255 symbols) can skip invalid-code checks without losing safety.

## Usage

```rust
use presst::{Compressor, Decompressor, Dictionary};

let dict = Dictionary::from_bytes(&std::fs::read("urls.pdict")?)?;

let compressor = Compressor::new(&dict)?;   // build the tables once
let frame = compressor.compress(b"https://example.com/index.html");

let decompressor = Decompressor::new(&dict)?;
assert_eq!(decompressor.decompress(&frame)?, b"https://example.com/index.html");

// Columnar use: many strings, each independently decodable.
let batch = compressor.compress_strings(["https://a.example", "https://b.example"]);
let mut out = Vec::new();
decompressor.decompress_string(&batch, 1, &mut out)?;
```

## Design

- **Code schemes.** The format supports several code widths. `U8` is
  implemented and produces exactly FSST's code stream (255 symbols of up to 8
  bytes, code 255 escapes a literal byte). Wider schemes with thousands of
  pretrained symbols are reserved in the format: since the dictionary is not
  stored per block, a large dictionary costs no space.
- **Frames** are a header plus independently decodable blocks with a raw
  fallback for incompressible data and an optional content checksum. See
  [`docs/FORMAT.md`](docs/FORMAT.md).
- **Safety.** The API is safe; small `unsafe` hot loops are tested against a
  slow reference implementation with property tests, fuzzing and Miri.
  Decoders never panic on malformed input.
- **Portable core.** The `presst` crate is `no_std` + `alloc`.

## Crates

| Crate | Purpose |
|---|---|
| `presst` | Core library: dictionaries, codec, frames, string batches |
| `presst-train` | Offline dictionary trainer |
| `presst-cli` | `presst` command-line tool: `train`, `compress`, `decompress`, `inspect`, `eval` |
| `presst-bench` | Benchmarks and evaluation against FSST, LZ4 and zstd (not published) |

## Roadmap

- [x] Core U8 codec, frames, string batches, golden files, fuzz targets
- [x] Trainer: FSST-faithful baseline, large-sample training, local search
- [x] CLI: `train`, `compress`, `decompress`, `inspect`, `eval`
- [x] Benchmarks against per-block FSST, LZ4, zstd and zstd with trained dictionaries
- [ ] Faster encoder (currently ~400 MB/s)
- [ ] Built-in dictionaries for common data (URLs, logs, TPC-H text)
- [ ] Wide code schemes (`U16`, `Hybrid8x16`) and multi-dictionary frames
- [ ] GPU backend (CUDA)

## Command line

```sh
presst train data/*.txt --lines -o urls.pdict     # train on sample data (one string per line)
presst compress -d urls.pdict input.txt -o input.prs
presst decompress -d urls.pdict input.prs -o input.txt
presst inspect urls.pdict --symbols
presst eval -d urls.pdict held-out.txt --lines   # ratio and speed on other data
```

## License

MIT
