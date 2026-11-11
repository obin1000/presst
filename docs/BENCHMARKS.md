# Benchmarks

Held-out evaluation on the 23 datasets of the FSST paper's
[dbtext corpus](https://github.com/cwida/fsst/tree/master/paper/dbtext)
(`cargo xtask corpus`). For each dataset, the first half of the lines trains
every trained component (presst dictionaries, the FSST table trained once, zstd
dictionaries) and the second half is the test data, so nothing is evaluated on
its own training data. Up to 4 MiB of test data per dataset.

Scenarios:

- **blocks**: the test data cut into independent blocks of the given size, each
  compressed on its own. Dictionaries for this scenario are trained on text
  (4 KiB windows, newlines included).
- **individual strings**: every line compressed on its own, as in a string
  column with random access. Dictionaries are trained on lines.

Codecs:

- **presst (extended)**: presst with a dictionary from the extended trainer.
- **presst (fsst-trained)**: presst with a dictionary from the FSST-faithful
  trainer, trained once on the training half.
- **fsst (per block)**: [fsst-rs](https://crates.io/crates/fsst-rs) with a table
  trained for every block; the serialized table is counted in the output.
- **fsst (trained once)**: fsst-rs with one table trained on the training half.
- **lz4** (lz4_flex), **zstd** -1/-3, and **zstd -3 with a trained dictionary**
  (requested maximum 2 KiB or 110 KiB; contexts and dictionaries are loaded once).

Ratio is total input / total output over all datasets; throughput is total
input / total time. Single thread; Intel Core i7-7740X (4.3 GHz), Windows,
rustc 1.99.1, `lto = "fat"`, generic x86-64 target (no `target-cpu=native`).

Reproduce:

```sh
cargo xtask corpus
cargo run --release -p presst-bench --bin presst-eval -- corpora/data/dbtext/* \
    --sizes 128,1024,65536,1048576 --min-secs 0.1 --json results.jsonl
cargo xtask report results.jsonl
```

## Results

**128 B blocks**

| Codec | Ratio | Compress MB/s | Decompress MB/s |
|---|---:|---:|---:|
| fsst (per block) | 0.60 | 0 | 243 |
| fsst (trained once) | 2.06 | 264 | 2313 |
| lz4 | 1.07 | 270 | 2970 |
| presst (extended) | 2.26 | 393 | 2584 |
| presst (fsst-trained) | 2.13 | 365 | 2451 |
| zstd -1 | 1.22 | 42 | 92 |
| zstd -3 | 1.22 | 40 | 91 |
| zstd -3 + dict (max 110 KiB) | 1.81 | 95 | 314 |
| zstd -3 + dict (max 2 KiB) | 1.66 | 89 | 319 |

**1 KiB blocks**

| Codec | Ratio | Compress MB/s | Decompress MB/s |
|---|---:|---:|---:|
| fsst (per block) | 1.05 | 1 | 713 |
| fsst (trained once) | 2.09 | 309 | 3302 |
| lz4 | 1.29 | 497 | 2933 |
| presst (extended) | 2.31 | 423 | 3904 |
| presst (fsst-trained) | 2.17 | 390 | 3604 |
| zstd -1 | 1.91 | 150 | 392 |
| zstd -3 | 1.90 | 134 | 361 |
| zstd -3 + dict (max 110 KiB) | 2.49 | 151 | 541 |
| zstd -3 + dict (max 2 KiB) | 2.26 | 139 | 515 |

**64 KiB blocks**

| Codec | Ratio | Compress MB/s | Decompress MB/s |
|---|---:|---:|---:|
| fsst (per block) | 2.05 | 39 | 3324 |
| fsst (trained once) | 2.10 | 324 | 3519 |
| lz4 | 1.59 | 544 | 2735 |
| presst (extended) | 2.31 | 431 | 4254 |
| presst (fsst-trained) | 2.17 | 397 | 3913 |
| zstd -1 | 2.51 | 289 | 992 |
| zstd -3 | 2.58 | 215 | 917 |
| zstd -3 + dict (max 110 KiB) | 2.71 | 142 | 722 |
| zstd -3 + dict (max 2 KiB) | 2.51 | 171 | 787 |

**1 MiB blocks**

| Codec | Ratio | Compress MB/s | Decompress MB/s |
|---|---:|---:|---:|
| fsst (per block) | 2.09 | 176 | 3529 |
| fsst (trained once) | 2.10 | 314 | 3521 |
| lz4 | 1.61 | 514 | 2726 |
| presst (extended) | 2.31 | 431 | 4273 |
| presst (fsst-trained) | 2.17 | 398 | 3919 |
| zstd -1 | 2.58 | 305 | 1092 |
| zstd -3 | 2.73 | 182 | 881 |
| zstd -3 + dict (max 110 KiB) | 2.72 | 157 | 878 |
| zstd -3 + dict (max 2 KiB) | 2.52 | 190 | 828 |

**individual strings**

| Codec | Ratio | Compress MB/s | Decompress MB/s |
|---|---:|---:|---:|
| fsst (trained once) | 2.07 | 158 | 1106 |
| lz4 | 0.95 | 80 | 1506 |
| presst (extended) | 2.29 | 345 | 1340 |
| presst (fsst-trained) | 2.14 | 321 | 1245 |
| zstd -1 | 0.76 | 55 | 219 |
| zstd -3 | 0.76 | 53 | 220 |
| zstd -3 + dict (max 110 KiB) | 0.89 | 34 | 173 |
| zstd -3 + dict (max 2 KiB) | 0.87 | 35 | 179 |

