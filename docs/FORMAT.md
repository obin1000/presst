# presst binary formats

Status: **unstable** (version 1, pre-1.0). Readers reject unknown versions. All
integers are little-endian. `xxh3` means XXH3-64 with seed 0.

There are three formats:

| Format | Magic | Purpose |
|---|---|---|
| Dictionary (`.pdict`) | `PRSD` | A pretrained symbol table |
| Frame (`.prs`) | `PRST` | A compressed byte stream |
| String batch (`.prsc`) | `PRSC` | Many individually compressed strings |

## Code schemes

A scheme defines how symbols map to codes. Its wire value appears in
dictionaries and frames.

| Value | Name | Status |
|---|---|---|
| 0 | `U8` | Implemented |
| 1 | `U16` | Reserved |
| 2 | `Hybrid8x16` | Reserved |
| 3 | `U12` | Reserved |

### U8 (FSST-compatible)

- A dictionary holds at most 255 symbols, each 1 to 8 bytes long. Symbol `i` is code `i`.
- The code stream is a sequence of bytes:
  - `c` in `0..len`: emit symbol `c`.
  - `255`: escape; the next byte is emitted literally.
  - Any other value is invalid.
- This is the same code stream FSST produces, so an FSST decoder with the same symbol table can decode it.

**Dictionary constraints:**
- No duplicate symbols.
- At most one symbol of 3 or more bytes per distinct 3-byte prefix. This lets the encoder find the only long candidate at each position with a single hash lookup.

**Encoding (normative):** greedy longest match.
- At each position, emit the code of the longest symbol that matches the remaining input.
- If none matches, emit `255` followed by the byte.
- Given the constraints, all conforming encoders produce identical output (the `reference` module is the oracle).

## Dictionary (`.pdict`)

```
offset  size      field
0       4         magic "PRSD"
4       2         format version = 1
6       1         scheme
7       1         max symbol length (8 for U8)
8       2         flags = 0 (nonzero is rejected)
10      4         symbol count n
14      4         metadata length m
18      m         metadata (TLV records)
18+m    n         symbol lengths, in code order
...     sum(lens) symbol bytes, concatenated in code order
end-8   8         xxh3 of all preceding bytes
```

**Metadata TLV record:** `tag: u8`, `len: u16`, `len` bytes. Readers skip unknown tags.

| Tag | Content |
|---|---|
| 1 | Name (UTF-8) |
| 2 | Trainer name and version (UTF-8) |
| 3 | Training seed (u64) |
| 4 | Corpus hash (u64) |

**Dictionary id:** `xxh3(scheme ‖ max_sym_len ‖ n as u32 ‖ lens ‖ symbol bytes)`. Metadata is excluded, so changing it keeps the id.

## Frame (`.prs`)

```
offset  size  field
0       4     magic "PRST"
4       1     format version = 1
5       1     flags
6       1     scheme
7       1     block size log2 (10..=26)
8       8     dictionary id
16      8     content size (only if flag CONTENT_SIZE)
        ...   blocks, until a block with the LAST bit
        8     xxh3 of the decompressed content (only if flag CHECKSUM)
```

| Flag | Bit | Meaning |
|---|---|---|
| `CHECKSUM` | 0 | Content checksum follows the last block |
| `CONTENT_SIZE` | 1 | Content size stored in the header |
| `DICT_INLINE` | 2 | Reserved: dictionary stored in the frame |
| `MULTI_DICT` | 3 | Reserved: several dictionaries, one chosen per block |

Readers reject frames with flags they don't support.

### Blocks

```
size  field
4     bits 0..29: csize (payload length)
      bit 29:     LAST
      bits 30..32: type (0 = coded, 1 = raw, 2-3 reserved)
4     dsize (decompressed length)
csize payload
```

- A **coded** block's payload is a code stream that decodes to exactly `dsize` bytes.
- A **raw** block stores the data as is; `csize == dsize`.

**Decoder checks:**
- `dsize <= block size`.
- For coded blocks, `dsize <= csize * max_expansion` (8 for U8). This bounds memory use by the input size.
- The sum of `dsize` equals the content size, if stored.
- No bytes after the frame.

**Writers:**
- Every block except the last holds exactly one block size of data.
- A block that doesn't get smaller when coded is stored raw.
- An empty input is one raw, empty block with the LAST bit set.

## String batch (`.prsc`)

```
size        field
4           magic "PRSC"
1           version = 1
8           dictionary id
4           string count n
4 * (n+1)   offsets into the code bytes (non-decreasing; first 0, last = code length)
rest        code bytes: the U8 code streams of all strings, concatenated
```

String `i` is the code stream `codes[offsets[i]..offsets[i+1]]`.
