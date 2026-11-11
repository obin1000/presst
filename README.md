# presst

**PREtrained Static Symbol Table compression**: FSST-style string compression
with symbol tables that are trained offline, once, instead of per block.

> Status: early development. The core codec works and is heavily tested; the
> trainer, CLI and benchmarks are in progress. The binary format is not stable
> before 1.0.

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
| `presst-cli` | `presst` command-line tool |
| `presst-bench` | Benchmarks and evaluation against FSST, LZ4 and zstd (not published) |

## Roadmap

- [x] Core U8 codec, frames, string batches, golden files, fuzz targets
- [ ] Trainer: FSST-faithful baseline, large-sample training, local search
- [ ] CLI: `train`, `compress`, `decompress`, `inspect`, `eval`
- [ ] Benchmarks against per-block FSST, LZ4, zstd and zstd with trained dictionaries
- [ ] Wide code schemes (`U16`, `Hybrid8x16`) and multi-dictionary frames
- [ ] GPU backend (CUDA)

## License

MIT
