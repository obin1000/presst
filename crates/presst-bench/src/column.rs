//! Database-style string column benchmark: columns split into row groups of
//! 122,880 values, scanned vector by vector (2,048 values), plus point
//! lookups of single rows. Modelled on DuckDB's storage layout.

use std::time::Instant;

use presst::{Compressor, Decompressor, Dictionary};

/// Values per row group (DuckDB's default).
pub const ROW_GROUP: usize = 122_880;
/// Values per vector (DuckDB's default).
pub const VECTOR: usize = 2048;
/// Maximum seconds per timed lookup run.
const LOOKUP_BUDGET_SECS: f64 = 0.5;

/// One compressed row group.
///
/// `lens` (value lengths) is part of every codec's layout, so it is excluded
/// from the compressed size: only `data` counts.
pub struct Encoded {
    /// Compressed bytes, including any per-row-group table.
    pub data: Vec<u8>,
    /// Codec-specific offsets into `data`: per value for per-value codecs,
    /// per vector for per-vector codecs.
    pub offsets: Vec<u32>,
    /// Uncompressed length of every value.
    pub lens: Vec<u32>,
}

/// A string column codec.
pub trait ColumnCodec: Sync {
    /// Display name.
    fn name(&self) -> String;
    /// Size of a shared dictionary or table, stored once per column (0 if none).
    fn shared_bytes(&self) -> usize {
        0
    }
    /// Compresses one row group.
    fn encode(&self, values: &[&[u8]]) -> Encoded;
    /// Appends values `range` (within one vector) to `out`.
    fn scan_vector(&self, enc: &Encoded, vector: usize, out: &mut Vec<u8>);
    /// Appends value `row` to `out`.
    fn get(&self, enc: &Encoded, row: usize, out: &mut Vec<u8>);
}

fn lens_of(values: &[&[u8]]) -> Vec<u32> {
    values.iter().map(|v| v.len() as u32).collect()
}

fn vector_range(n: usize, vector: usize) -> std::ops::Range<usize> {
    vector * VECTOR..((vector + 1) * VECTOR).min(n)
}

fn decoded_len(lens: &[u32], range: std::ops::Range<usize>) -> usize {
    lens[range].iter().map(|&l| l as usize).sum()
}

// ---------------------------------------------------------------- presst

/// presst with one pretrained dictionary per column; values coded individually.
pub struct PresstColumn {
    label: String,
    comp: Compressor,
    dec: Decompressor,
    dict_bytes: usize,
}

impl PresstColumn {
    /// Prepares the tables for `dict`.
    ///
    /// # Panics
    /// Panics if the dictionary's scheme is not supported.
    #[must_use]
    pub fn new(label: &str, dict: &Dictionary) -> Self {
        Self {
            label: label.to_owned(),
            comp: Compressor::new(dict).expect("supported scheme"),
            dec: Decompressor::new(dict).expect("supported scheme"),
            dict_bytes: dict.to_bytes().len(),
        }
    }
}

impl ColumnCodec for PresstColumn {
    fn name(&self) -> String {
        self.label.clone()
    }
    fn shared_bytes(&self) -> usize {
        self.dict_bytes
    }
    fn encode(&self, values: &[&[u8]]) -> Encoded {
        let mut data = Vec::new();
        let mut offsets = Vec::with_capacity(values.len() + 1);
        offsets.push(0);
        for v in values {
            self.comp.compress_block_into(v, &mut data);
            offsets.push(data.len() as u32);
        }
        Encoded {
            data,
            offsets,
            lens: lens_of(values),
        }
    }
    fn scan_vector(&self, enc: &Encoded, vector: usize, out: &mut Vec<u8>) {
        // The per-value code streams of a vector are contiguous and decode to
        // the concatenated values, so one call decodes the whole vector.
        let r = vector_range(enc.lens.len(), vector);
        let codes = &enc.data[enc.offsets[r.start] as usize..enc.offsets[r.end] as usize];
        self.dec
            .decompress_block_into(codes, decoded_len(&enc.lens, r), out)
            .expect("valid");
    }
    fn get(&self, enc: &Encoded, row: usize, out: &mut Vec<u8>) {
        let codes = &enc.data[enc.offsets[row] as usize..enc.offsets[row + 1] as usize];
        self.dec
            .decompress_block_into(codes, enc.lens[row] as usize, out)
            .expect("valid");
    }
}

// ---------------------------------------------------------------- FSST

fn fsst_symbols(data: &[u8]) -> (Vec<fsst::Symbol>, Vec<u8>, usize) {
    let n = data[0] as usize;
    let lens = data[1..=n].to_vec();
    let mut pos = 1 + n;
    let symbols = lens
        .iter()
        .map(|&l| {
            let mut b = [0u8; 8];
            b[..l as usize].copy_from_slice(&data[pos..pos + l as usize]);
            pos += l as usize;
            fsst::Symbol::from_slice(&b)
        })
        .collect();
    (symbols, lens, pos)
}

fn fsst_decode(dec: &fsst::Decompressor<'_>, codes: &[u8], len: usize, out: &mut Vec<u8>) {
    let cap = dec.max_decompression_capacity(codes).max(len);
    out.reserve(cap);
    let start = out.len();
    let n = dec.decompress_into(codes, &mut out.spare_capacity_mut()[..cap]);
    assert_eq!(n, len, "fsst decoded size");
    // SAFETY: `decompress_into` initialized the first `n` bytes of the spare capacity.
    unsafe { out.set_len(start + n) };
}

/// FSST as DuckDB uses it: a table trained per row group and stored with it.
pub struct FsstRowGroup;

impl ColumnCodec for FsstRowGroup {
    fn name(&self) -> String {
        "fsst (table per row group)".into()
    }
    fn encode(&self, values: &[&[u8]]) -> Encoded {
        let comp = fsst::Compressor::train(&values.to_vec());
        let lens = comp.symbol_lengths();
        let mut data = vec![lens.len() as u8];
        data.extend_from_slice(lens);
        for (s, &l) in comp.symbol_table().iter().zip(lens) {
            data.extend_from_slice(&s.to_u64().to_le_bytes()[..l as usize]);
        }
        let mut offsets = Vec::with_capacity(values.len() + 1);
        offsets.push(data.len() as u32);
        for v in values {
            data.extend_from_slice(&comp.compress(v));
            offsets.push(data.len() as u32);
        }
        Encoded {
            data,
            offsets,
            lens: lens_of(values),
        }
    }
    fn scan_vector(&self, enc: &Encoded, vector: usize, out: &mut Vec<u8>) {
        // The table is parsed per vector here; DuckDB keeps it per segment,
        // so this slightly favours nothing but simplicity (parsing is cheap).
        let (symbols, lens, _) = fsst_symbols(&enc.data);
        let dec = fsst::Decompressor::new(&symbols, &lens);
        let r = vector_range(enc.lens.len(), vector);
        let codes = &enc.data[enc.offsets[r.start] as usize..enc.offsets[r.end] as usize];
        fsst_decode(&dec, codes, decoded_len(&enc.lens, r), out);
    }
    fn get(&self, enc: &Encoded, row: usize, out: &mut Vec<u8>) {
        let (symbols, lens, _) = fsst_symbols(&enc.data);
        let dec = fsst::Decompressor::new(&symbols, &lens);
        let codes = &enc.data[enc.offsets[row] as usize..enc.offsets[row + 1] as usize];
        fsst_decode(&dec, codes, enc.lens[row] as usize, out);
    }
}

/// An FSST table trained once (fsst-rs's own trainer) and shared by the column.
pub struct FsstShared {
    comp: fsst::Compressor,
}

impl FsstShared {
    /// Trains the table on `samples`.
    #[must_use]
    pub fn train(samples: &[&[u8]]) -> Self {
        Self {
            comp: fsst::Compressor::train(&samples.to_vec()),
        }
    }
}

impl ColumnCodec for FsstShared {
    fn name(&self) -> String {
        "fsst (shared table)".into()
    }
    fn shared_bytes(&self) -> usize {
        1 + self.comp.symbol_lengths().len() + self.comp.symbol_lengths().iter().map(|&l| l as usize).sum::<usize>()
    }
    fn encode(&self, values: &[&[u8]]) -> Encoded {
        let mut data = Vec::new();
        let mut offsets = Vec::with_capacity(values.len() + 1);
        offsets.push(0);
        for v in values {
            data.extend_from_slice(&self.comp.compress(v));
            offsets.push(data.len() as u32);
        }
        Encoded {
            data,
            offsets,
            lens: lens_of(values),
        }
    }
    fn scan_vector(&self, enc: &Encoded, vector: usize, out: &mut Vec<u8>) {
        let r = vector_range(enc.lens.len(), vector);
        let codes = &enc.data[enc.offsets[r.start] as usize..enc.offsets[r.end] as usize];
        fsst_decode(&self.comp.decompressor(), codes, decoded_len(&enc.lens, r), out);
    }
    fn get(&self, enc: &Encoded, row: usize, out: &mut Vec<u8>) {
        let codes = &enc.data[enc.offsets[row] as usize..enc.offsets[row + 1] as usize];
        fsst_decode(&self.comp.decompressor(), codes, enc.lens[row] as usize, out);
    }
}

// ---------------------------------------------------------------- per vector

/// A general-purpose block codec applied to each vector of 2,048 values
/// (concatenated), like a Parquet page or DuckDB block compression.
pub struct PerVector<C> {
    inner: C,
}

impl<C> PerVector<C> {
    /// Wraps a block codec.
    pub fn new(inner: C) -> Self {
        Self { inner }
    }
}

impl<C: crate::BlockCodec + SharedBytes> ColumnCodec for PerVector<C> {
    fn name(&self) -> String {
        format!("{} /vector", self.inner.name())
    }
    fn shared_bytes(&self) -> usize {
        self.inner.shared()
    }
    fn encode(&self, values: &[&[u8]]) -> Encoded {
        let mut data = Vec::new();
        let mut offsets = vec![0u32];
        let mut buf = Vec::new();
        for chunk in values.chunks(VECTOR) {
            buf.clear();
            for v in chunk {
                buf.extend_from_slice(v);
            }
            self.inner.compress(&buf, &mut data);
            offsets.push(data.len() as u32);
        }
        Encoded {
            data,
            offsets,
            lens: lens_of(values),
        }
    }
    fn scan_vector(&self, enc: &Encoded, vector: usize, out: &mut Vec<u8>) {
        let r = vector_range(enc.lens.len(), vector);
        let block = &enc.data[enc.offsets[vector] as usize..enc.offsets[vector + 1] as usize];
        self.inner.decompress(block, decoded_len(&enc.lens, r), out);
    }
    fn get(&self, enc: &Encoded, row: usize, out: &mut Vec<u8>) {
        // A point lookup has to decompress the whole vector.
        let vector = row / VECTOR;
        let r = vector_range(enc.lens.len(), vector);
        let mut tmp = Vec::with_capacity(decoded_len(&enc.lens, r.clone()));
        self.scan_vector(enc, vector, &mut tmp);
        let start = decoded_len(&enc.lens, r.start..row);
        out.extend_from_slice(&tmp[start..start + enc.lens[row] as usize]);
    }
}

/// Size of a block codec's shared dictionary.
pub trait SharedBytes {
    /// Bytes stored once per column.
    fn shared(&self) -> usize;
}

impl SharedBytes for crate::Lz4 {
    fn shared(&self) -> usize {
        0
    }
}

impl SharedBytes for crate::Zstd {
    fn shared(&self) -> usize {
        self.dict_len()
    }
}

// ---------------------------------------------------------------- measurement

/// Results for one codec on one column.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ColumnResult {
    /// Codec name.
    pub codec: String,
    /// Number of values.
    pub values: usize,
    /// Uncompressed value bytes.
    pub input_bytes: u64,
    /// Compressed bytes (excluding value lengths, which every codec stores).
    pub compressed_bytes: u64,
    /// `input_bytes / compressed_bytes`.
    pub ratio: f64,
    /// Shared dictionary/table bytes, stored once per column.
    pub shared_bytes: usize,
    /// Compression throughput, MB/s of input.
    pub compress_mbps: f64,
    /// Full scan (all vectors), MB/s of output.
    pub scan_mbps: f64,
    /// Mean point-lookup latency in nanoseconds.
    pub lookup_ns: f64,
    /// Lookups in the timed run (fewer than requested if the time budget ran out).
    pub lookups: usize,
}

/// Compresses `values` in row groups, verifies scans and lookups, and times
/// compression, full scans and `lookups` random point lookups. Each timing is
/// the median of `runs` runs.
///
/// # Panics
/// Panics if a value does not round-trip.
pub fn measure_column(codec: &dyn ColumnCodec, values: &[&[u8]], runs: usize, lookups: usize) -> ColumnResult {
    let input: u64 = values.iter().map(|v| v.len() as u64).sum();
    let groups: Vec<&[&[u8]]> = values.chunks(ROW_GROUP).collect();
    let runs = runs.max(1);

    let mut encoded = Vec::new();
    let compress = median(runs, || {
        let t = Instant::now();
        encoded = groups.iter().map(|g| codec.encode(g)).collect::<Vec<_>>();
        t.elapsed().as_secs_f64()
    });

    // Verify the scan once, outside the timing.
    let mut out = Vec::with_capacity(input as usize + 64);
    let mut row = 0;
    for (g, enc) in groups.iter().zip(&encoded) {
        for v in 0..g.len().div_ceil(VECTOR) {
            out.clear();
            codec.scan_vector(enc, v, &mut out);
            let expected: Vec<u8> = g[vector_range(g.len(), v)].concat();
            assert!(out == expected, "{}: scan mismatch at row {row}", codec.name());
            row += VECTOR;
        }
    }
    let scan = median(runs, || {
        let t = Instant::now();
        for enc in &encoded {
            for v in 0..enc.lens.len().div_ceil(VECTOR) {
                out.clear();
                codec.scan_vector(enc, v, &mut out);
            }
        }
        t.elapsed().as_secs_f64()
    });

    // Point lookups at deterministic pseudo-random rows.
    let n = values.len();
    let rows: Vec<usize> = (0..lookups)
        .map(|i| ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11) as usize % n.max(1))
        .collect();
    for &r in rows.iter().take(1000) {
        out.clear();
        codec.get(&encoded[r / ROW_GROUP], r % ROW_GROUP, &mut out);
        assert!(out == values[r], "{}: lookup mismatch at row {r}", codec.name());
    }
    // Lookups run until all rows are done or a time budget is used up, so
    // slow codecs (a vector decompression per lookup) stay affordable.
    let mut done = rows.len();
    let lookup = median(runs, || {
        let t = Instant::now();
        let mut i = 0;
        while i < rows.len() {
            let r = rows[i];
            out.clear();
            codec.get(&encoded[r / ROW_GROUP], r % ROW_GROUP, &mut out);
            i += 1;
            if i % 256 == 0 && t.elapsed().as_secs_f64() > LOOKUP_BUDGET_SECS {
                break;
            }
        }
        done = i;
        t.elapsed().as_secs_f64() / i.max(1) as f64
    });

    let compressed: u64 = encoded.iter().map(|e| e.data.len() as u64).sum();
    ColumnResult {
        codec: codec.name(),
        values: n,
        input_bytes: input,
        compressed_bytes: compressed,
        ratio: input as f64 / compressed.max(1) as f64,
        shared_bytes: codec.shared_bytes(),
        compress_mbps: input as f64 / 1e6 / compress,
        scan_mbps: input as f64 / 1e6 / scan,
        lookup_ns: lookup * 1e9,
        lookups: done,
    }
}

fn median(runs: usize, mut f: impl FnMut() -> f64) -> f64 {
    f(); // warm-up
    let mut t: Vec<f64> = (0..runs).map(|_| f()).collect();
    t.sort_by(f64::total_cmp);
    t[t.len() / 2].max(1e-9)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Lz4, Zstd};
    use presst::Scheme;

    fn column(n: usize) -> Vec<Vec<u8>> {
        (0..n)
            .map(|i| format!("https://example.com/item/{i}?q={}", i % 7).into_bytes())
            .collect()
    }

    #[test]
    fn codecs_roundtrip_at_boundaries() {
        let dict =
            Dictionary::from_symbols(Scheme::U8, [&b"https://"[..], b"example", b".com/", b"item/", b"?q="]).unwrap();
        let samples = column(500);
        let sample_refs: Vec<&[u8]> = samples.iter().map(Vec::as_slice).collect();
        let codecs: Vec<Box<dyn ColumnCodec>> = vec![
            Box::new(PresstColumn::new("presst", &dict)),
            Box::new(FsstRowGroup),
            Box::new(FsstShared::train(&sample_refs)),
            Box::new(PerVector::new(Lz4)),
            Box::new(PerVector::new(Zstd::new(3))),
        ];
        for n in [0usize, 1, VECTOR - 1, VECTOR + 1, ROW_GROUP + 1] {
            let col = column(n);
            let refs: Vec<&[u8]> = col.iter().map(Vec::as_slice).collect();
            for c in &codecs {
                // `measure_column` verifies scans and lookups.
                let r = measure_column(c.as_ref(), &refs, 1, if n == 0 { 0 } else { 100 });
                assert_eq!(r.values, n, "{}", c.name());
            }
        }
    }
}
