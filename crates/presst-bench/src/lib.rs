//! Evaluation harness for presst: codecs behind one interface, so presst can be
//! compared with per-block FSST, LZ4, zstd and zstd with trained dictionaries
//! on the same held-out data.

#![allow(missing_debug_implementations)]

pub mod column;
pub mod vals;

use std::mem::MaybeUninit;

use presst::{Compressor, Decompressor, Dictionary};

/// A codec that compresses independent blocks (or single strings).
pub trait BlockCodec: Sync {
    /// Display name.
    fn name(&self) -> String;
    /// Appends the compressed form of `block` to `out`.
    fn compress(&self, block: &[u8], out: &mut Vec<u8>);
    /// Appends the decompressed form of `data` (of original length `len`) to `out`.
    fn decompress(&self, data: &[u8], len: usize, out: &mut Vec<u8>);
}

/// presst with a pretrained dictionary: raw code streams, no framing.
pub struct Presst {
    label: String,
    comp: Compressor,
    dec: Decompressor,
}

impl Presst {
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
        }
    }
}

impl BlockCodec for Presst {
    fn name(&self) -> String {
        self.label.clone()
    }
    fn compress(&self, block: &[u8], out: &mut Vec<u8>) {
        self.comp.compress_block_into(block, out);
    }
    fn decompress(&self, data: &[u8], len: usize, out: &mut Vec<u8>) {
        self.dec.decompress_block_into(data, len, out).expect("valid stream");
    }
}

/// FSST as normally used: a table trained for every block and stored with it.
pub struct FsstPerBlock;

impl BlockCodec for FsstPerBlock {
    fn name(&self) -> String {
        "fsst (per block)".into()
    }
    fn compress(&self, block: &[u8], out: &mut Vec<u8>) {
        let comp = fsst::Compressor::train(&vec![block]);
        // Serialized table: count, lengths, symbol bytes (as FSST stores it,
        // minus FSST's own small header).
        let lens = comp.symbol_lengths();
        out.push(lens.len() as u8);
        out.extend_from_slice(lens);
        for (sym, &len) in comp.symbol_table().iter().zip(lens) {
            out.extend_from_slice(&sym.to_u64().to_le_bytes()[..len as usize]);
        }
        out.extend_from_slice(&comp.compress(block));
    }
    fn decompress(&self, data: &[u8], len: usize, out: &mut Vec<u8>) {
        let n = data[0] as usize;
        let lens = &data[1..=n];
        let mut pos = 1 + n;
        let symbols: Vec<fsst::Symbol> = lens
            .iter()
            .map(|&l| {
                let mut b = [0u8; 8];
                b[..l as usize].copy_from_slice(&data[pos..pos + l as usize]);
                pos += l as usize;
                fsst::Symbol::from_slice(&b)
            })
            .collect();
        decompress_fsst(&fsst::Decompressor::new(&symbols, lens), &data[pos..], len, out);
    }
}

/// An FSST table trained once (with FSST's own trainer) and reused for every
/// block: isolates the effect of pretraining from presst's better trainer.
pub struct FsstOnce {
    comp: fsst::Compressor,
}

impl FsstOnce {
    /// Trains the table on `samples`.
    #[must_use]
    pub fn train(samples: &[&[u8]]) -> Self {
        Self {
            comp: fsst::Compressor::train(&samples.to_vec()),
        }
    }
}

impl BlockCodec for FsstOnce {
    fn name(&self) -> String {
        "fsst (trained once)".into()
    }
    fn compress(&self, block: &[u8], out: &mut Vec<u8>) {
        out.extend_from_slice(&self.comp.compress(block));
    }
    fn decompress(&self, data: &[u8], len: usize, out: &mut Vec<u8>) {
        decompress_fsst(&self.comp.decompressor(), data, len, out);
    }
}

fn decompress_fsst(dec: &fsst::Decompressor<'_>, data: &[u8], len: usize, out: &mut Vec<u8>) {
    let cap = dec.max_decompression_capacity(data).max(len);
    out.reserve(cap);
    let start = out.len();
    let spare: &mut [MaybeUninit<u8>] = &mut out.spare_capacity_mut()[..cap];
    let n = dec.decompress_into(data, spare);
    assert_eq!(n, len, "fsst decoded size");
    // SAFETY: `decompress_into` initialized the first `n` bytes of the spare capacity.
    unsafe { out.set_len(start + n) };
}

/// LZ4 block format (`lz4_flex`).
pub struct Lz4;

impl BlockCodec for Lz4 {
    fn name(&self) -> String {
        "lz4".into()
    }
    fn compress(&self, block: &[u8], out: &mut Vec<u8>) {
        out.extend_from_slice(&lz4_flex::block::compress(block));
    }
    fn decompress(&self, data: &[u8], len: usize, out: &mut Vec<u8>) {
        let start = out.len();
        out.resize(start + len, 0);
        let n = lz4_flex::block::decompress_into(data, &mut out[start..]).expect("valid lz4");
        assert_eq!(n, len);
    }
}

/// zstd, optionally with a trained dictionary.
pub struct Zstd {
    level: i32,
    dict: Option<Vec<u8>>,
    /// Requested maximum dictionary size (zstd may build a smaller one).
    max_dict: usize,
}

impl Zstd {
    /// zstd at `level` without a dictionary.
    #[must_use]
    pub fn new(level: i32) -> Self {
        Self {
            level,
            dict: None,
            max_dict: 0,
        }
    }

    /// zstd at `level` with a dictionary trained on `samples` (at most
    /// `max_size` bytes). Returns `None` if zstd cannot train one.
    #[must_use]
    pub fn with_trained_dict(level: i32, samples: &[&[u8]], max_size: usize) -> Option<Self> {
        let dict = zstd::dict::from_samples(samples, max_size).ok()?;
        Some(Self {
            level,
            dict: Some(dict),
            max_dict: max_size,
        })
    }

    /// Size of the dictionary, if any.
    #[must_use]
    pub fn dict_len(&self) -> usize {
        self.dict.as_ref().map_or(0, Vec::len)
    }
}

thread_local! {
    static ZSTD: std::cell::RefCell<Option<(usize, zstd::bulk::Compressor<'static>, zstd::bulk::Decompressor<'static>)>> =
        const { std::cell::RefCell::new(None) };
}

impl Zstd {
    fn with_ctx<T>(
        &self,
        f: impl FnOnce(&mut zstd::bulk::Compressor<'static>, &mut zstd::bulk::Decompressor<'static>) -> T,
    ) -> T {
        // Contexts (with the dictionary loaded) are created once per thread
        // and codec, like a real application would.
        let key = std::ptr::from_ref(self) as usize;
        ZSTD.with(|cell| {
            let mut slot = cell.borrow_mut();
            if slot.as_ref().is_none_or(|(k, _, _)| *k != key) {
                let (c, d) = match &self.dict {
                    Some(dict) => (
                        zstd::bulk::Compressor::with_dictionary(self.level, dict).expect("zstd"),
                        zstd::bulk::Decompressor::with_dictionary(dict).expect("zstd"),
                    ),
                    None => (
                        zstd::bulk::Compressor::new(self.level).expect("zstd"),
                        zstd::bulk::Decompressor::new().expect("zstd"),
                    ),
                };
                *slot = Some((key, c, d));
            }
            let (_, c, d) = slot.as_mut().expect("initialized");
            f(c, d)
        })
    }
}

impl BlockCodec for Zstd {
    fn name(&self) -> String {
        match &self.dict {
            Some(_) => format!("zstd -{} +dict{}K", self.level, self.max_dict / 1024),
            None => format!("zstd -{}", self.level),
        }
    }
    fn compress(&self, block: &[u8], out: &mut Vec<u8>) {
        self.with_ctx(|c, _| {
            let bound = zstd::zstd_safe::compress_bound(block.len());
            let start = out.len();
            out.resize(start + bound, 0);
            let n = c.compress_to_buffer(block, &mut out[start..]).expect("zstd compress");
            out.truncate(start + n);
        });
    }
    fn decompress(&self, data: &[u8], len: usize, out: &mut Vec<u8>) {
        self.with_ctx(|_, d| {
            let start = out.len();
            out.resize(start + len, 0);
            let n = d
                .decompress_to_buffer(data, &mut out[start..])
                .expect("zstd decompress");
            assert_eq!(n, len);
        });
    }
}

/// Result of running one codec over a set of blocks.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Measurement {
    /// Codec name.
    pub codec: String,
    /// Uncompressed bytes.
    pub input_bytes: u64,
    /// Compressed bytes (including any per-block table).
    pub compressed_bytes: u64,
    /// `input_bytes / compressed_bytes`.
    pub ratio: f64,
    /// Compression throughput in MB/s of input.
    pub compress_mbps: f64,
    /// Decompression throughput in MB/s of output.
    pub decompress_mbps: f64,
}

/// Compresses and decompresses every block with `codec`, verifies the round
/// trip, and measures the best throughput over repeated runs of at least
/// `min_secs` each.
///
/// # Panics
/// Panics if a block does not round-trip.
pub fn measure(codec: &dyn BlockCodec, blocks: &[&[u8]], min_secs: f64) -> Measurement {
    let input: u64 = blocks.iter().map(|b| b.len() as u64).sum();
    let mut compressed: Vec<Vec<u8>> = vec![Vec::new(); blocks.len()];
    let compress_secs = best_time(min_secs, || {
        for (b, out) in blocks.iter().zip(compressed.iter_mut()) {
            out.clear();
            codec.compress(b, out);
        }
    });
    let mut out = Vec::with_capacity(input as usize + 64);
    let decompress_secs = best_time(min_secs, || {
        out.clear();
        for (c, b) in compressed.iter().zip(blocks) {
            codec.decompress(c, b.len(), &mut out);
        }
    });
    // Verify after timing, so verification does not distort the numbers.
    let mut pos = 0;
    for b in blocks {
        assert_eq!(&out[pos..pos + b.len()], *b, "{} round trip", codec.name());
        pos += b.len();
    }
    let csize: u64 = compressed.iter().map(|c| c.len() as u64).sum();
    Measurement {
        codec: codec.name(),
        input_bytes: input,
        compressed_bytes: csize,
        ratio: input as f64 / csize.max(1) as f64,
        compress_mbps: input as f64 / 1e6 / compress_secs,
        decompress_mbps: input as f64 / 1e6 / decompress_secs,
    }
}

fn best_time(min_secs: f64, mut f: impl FnMut()) -> f64 {
    let start = std::time::Instant::now();
    let mut best = f64::INFINITY;
    loop {
        let t = std::time::Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
        if start.elapsed().as_secs_f64() >= min_secs {
            return best.max(1e-9);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use presst::Scheme;

    fn sample() -> Vec<u8> {
        b"the quick brown fox jumps over the lazy dog\n".repeat(200)
    }

    #[test]
    fn all_codecs_roundtrip() {
        let data = sample();
        let blocks: Vec<&[u8]> = data.chunks(1000).collect();
        let dict = Dictionary::from_symbols(Scheme::U8, [&b"the "[..], b"quick", b" fox", b"o"]).unwrap();
        let codecs: Vec<Box<dyn BlockCodec>> = vec![
            Box::new(Presst::new("presst", &dict)),
            Box::new(FsstPerBlock),
            Box::new(FsstOnce::train(&blocks)),
            Box::new(Lz4),
            Box::new(Zstd::new(3)),
        ];
        for c in &codecs {
            // `measure` verifies the round trip.
            let m = measure(c.as_ref(), &blocks, 0.0);
            assert_eq!(m.input_bytes, data.len() as u64);
        }
    }

    /// presst's U8 code stream is FSST's: the reference FSST decoder (fsst-rs),
    /// loaded with the same symbols, decodes it.
    #[test]
    fn fsst_decodes_presst_u8_streams() {
        let data = sample();
        let dict = Dictionary::from_symbols(
            Scheme::U8,
            [&b"the "[..], b"quick", b" brown", b"fox", b"o", b"\n", b"jumps ov"],
        )
        .unwrap();
        let symbols: Vec<fsst::Symbol> = dict
            .symbols()
            .map(|s| {
                let mut b = [0u8; 8];
                b[..s.len()].copy_from_slice(s);
                fsst::Symbol::from_slice(&b)
            })
            .collect();
        let mut codes = Vec::new();
        Compressor::new(&dict).unwrap().compress_block_into(&data, &mut codes);
        let decoded = fsst::Decompressor::new(&symbols, dict.symbol_lens()).decompress(&codes);
        assert_eq!(decoded, data);
    }
}
