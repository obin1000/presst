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
per-block training and table overhead dominate. The price is that a dictionary
must be trained per column, on that column's own data (see Results).

## Results

204 real string columns (ClickBench, Loghub, Public BI), stored like a column
store (row groups, vectors, a dictionary layer for every codec). Every
dictionary is trained only on **older data or another table** of the same
column, never on the data it compresses. Single thread, Intel i7-7740X. Full
method, all tables and caveats: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

| Codec | String ratio | Compress | Scan | Point lookup |
|---|---:|---:|---:|---:|
| **presst** | **2.20** | **343 MB/s** | 3.1 GB/s | **43 ns** |
| FSST, table per row group (as in DuckDB) | 2.12 | 181 MB/s | 3.3 GB/s | 51 ns |
| FSST, one shared table | 1.94 | 214 MB/s | 2.9 GB/s | 45 ns |
| LZ4 per vector of 2,048 values | 2.54 | 480 MB/s | 2.8 GB/s | 6.6 µs |
| zstd -3 + dictionary per vector | 4.31 | 200 MB/s | 0.9 GB/s | 18.8 µs |

What this shows:

- **Versus FSST, presst compresses 1.9× faster at a slightly better ratio**,
  with the same random access, because no table is trained at ingest time.
- **Versus zstd, presst gives up about half the ratio** for 3× faster scans
  and ~440× faster point lookups.
- **Training on the column's own past is enough**: it reaches 96–99% of a
  dictionary trained on the test data itself (75% for logs, which drift), and
  removing every value seen during training changes the result no more than
  for any other codec. A dictionary holds at most ~2 KB, so it cannot memorize.
- **More offline data pays off**: 16 KiB of training data (FSST's sample) gives
  a string ratio of 1.63, 64 MiB gives 2.13 (+31%).
- **What does not work: generic dictionaries.** Dictionaries trained on other
  tables or datasets reach only 41–57% of the oracle. presst needs a dictionary
  per column (or per very similar data), retrained as the data drifts.

An earlier, smaller benchmark on the FSST paper's dbtext corpus (byte blocks
of 128 B to 1 MiB) is also in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

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
- [x] Column benchmark on real data with training-distance levels and stress tests
- [ ] Faster encoder and scan (FSST's decoder is still ~5% faster on columns)
- [ ] Automatic retraining and fallback when a dictionary ages
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
