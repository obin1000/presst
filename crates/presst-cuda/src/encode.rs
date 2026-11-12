//! Host side of GPU encoding: encode tables, work ranges, and assembling the
//! GPU's code streams into frames and string batches exactly like the CPU
//! writer does.

use presst::{Dictionary, FrameHeader, FrameOptions, Scheme, block_header};
use xxhash_rust::xxh3::xxh3_64;

const SLOT_BITS: u32 = 10;
const SLOTS: usize = 1 << SLOT_BITS;
const HASH_MUL: u32 = 0x9E37_79B1;

/// Size of the kernel's `EncodeTable`.
pub(crate) const ENCODE_TABLE_BYTES: usize = SLOTS * 8 + SLOTS + SLOTS + SLOTS * 4 + 256;

/// Must match `hash_slot` in `kernels/presst_encode.cuh`.
fn hash_slot(key: u32) -> usize {
    (key.wrapping_mul(HASH_MUL) >> (32 - SLOT_BITS)) as usize
}

/// The kernel's `EncodeTable` for `dict` (struct of arrays, little-endian).
///
/// Long symbols (3..8 bytes) are keyed by their first 3 bytes, 2-byte symbols
/// by both bytes; both tables use linear probing. A U8 dictionary has at most
/// 255 symbols, so the 1024-slot tables are at most a quarter full.
pub(crate) fn encode_table_bytes(dict: &Dictionary) -> Vec<u8> {
    debug_assert_eq!(dict.scheme(), Scheme::U8);
    let mut long_sym = vec![0u64; SLOTS];
    let mut long_len = vec![0u8; SLOTS];
    let mut long_code = vec![0u8; SLOTS];
    let mut two = vec![0u32; SLOTS];
    let mut byte_code = [255u8; 256];
    for (code, s) in dict.symbols().enumerate() {
        let code8 = code as u8;
        match s.len() {
            1 => byte_code[s[0] as usize] = code8,
            2 => {
                let key = u32::from(u16::from_le_bytes([s[0], s[1]]));
                let mut slot = hash_slot(key);
                while two[slot] != 0 {
                    slot = (slot + 1) & (SLOTS - 1);
                }
                two[slot] = key | (u32::from(code8) << 16) | (1 << 24);
            }
            len => {
                let mut word = [0u8; 8];
                word[..len].copy_from_slice(s);
                let sym = u64::from_le_bytes(word);
                let mut slot = hash_slot((sym & 0xFF_FFFF) as u32);
                while long_len[slot] != 0 {
                    slot = (slot + 1) & (SLOTS - 1);
                }
                long_sym[slot] = sym;
                long_len[slot] = len as u8;
                long_code[slot] = code8;
            }
        }
    }
    let mut out = Vec::with_capacity(ENCODE_TABLE_BYTES);
    for v in long_sym {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&long_len);
    out.extend_from_slice(&long_code);
    for v in two {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&byte_code);
    debug_assert_eq!(out.len(), ENCODE_TABLE_BYTES);
    out
}

/// One unit of encode work: the kernel's `Range`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Range {
    pub(crate) in_start: u64,
    /// Where the codes go in the scratch area: room for `2 * in_len` bytes,
    /// 8-byte aligned.
    pub(crate) out_start: u64,
    pub(crate) in_len: u32,
}

pub(crate) const RANGE_BYTES: usize = 24;

/// Ranges for consecutive pieces of the input with these lengths, and the
/// scratch size they need.
pub(crate) fn ranges(pieces: impl IntoIterator<Item = (u64, u32)>) -> (Vec<Range>, u64) {
    let mut scratch = 0u64;
    let ranges = pieces
        .into_iter()
        .map(|(in_start, in_len)| {
            let r = Range {
                in_start,
                out_start: scratch,
                in_len,
            };
            scratch += (2 * u64::from(in_len)).next_multiple_of(8);
            r
        })
        .collect();
    (ranges, scratch)
}

pub(crate) fn ranges_to_bytes(ranges: &[Range]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ranges.len() * RANGE_BYTES);
    for r in ranges {
        out.extend_from_slice(&r.in_start.to_le_bytes());
        out.extend_from_slice(&r.out_start.to_le_bytes());
        out.extend_from_slice(&r.in_len.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
    }
    out
}

/// How a frame's input is cut into encode ranges: per block, the ranges of its
/// splits (or the whole block without a split index).
pub(crate) struct FramePlan {
    pub(crate) options: FrameOptions,
    /// For each block: (input start, input length, index of its first range, number of ranges).
    pub(crate) blocks: Vec<(usize, usize, usize, usize)>,
    pub(crate) pieces: Vec<(u64, u32)>,
}

pub(crate) fn plan_frame(input_len: usize, options: FrameOptions) -> FramePlan {
    let block = options.block_size();
    let split = options.split_size_log2().map_or(block, |s| 1usize << s);
    let mut blocks = Vec::new();
    let mut pieces = Vec::new();
    let mut start = 0;
    while start < input_len {
        let len = block.min(input_len - start);
        let first = pieces.len();
        let mut s = 0;
        while s < len {
            let n = split.min(len - s);
            pieces.push(((start + s) as u64, n as u32));
            s += n;
        }
        blocks.push((start, len, first, pieces.len() - first));
        start += len;
    }
    FramePlan {
        options,
        blocks,
        pieces,
    }
}

/// Assembles the frame from the GPU's codes: `codes` holds the codes of all
/// ranges back to back, `lens` the length of each. Follows the CPU writer:
/// split index, raw fallback when coding does not save space, checksum.
pub(crate) fn assemble_frame(plan: &FramePlan, dict: &Dictionary, input: &[u8], codes: &[u8], lens: &[u32]) -> Vec<u8> {
    let options = plan.options;
    let header = FrameHeader {
        scheme: dict.scheme(),
        block_size_log2: options.block_size_log2(),
        dict_id: dict.id(),
        checksum: options.checksum,
        content_size: options.content_size.then_some(input.len() as u64),
        split_size_log2: options.split_size_log2(),
    };
    let mut out = Vec::with_capacity(input.len() + 64);
    header.write(&mut out);
    if input.is_empty() {
        out.extend_from_slice(&block_header(true, true, 0, 0));
    }
    let mut code_pos = 0usize;
    let splits = options.split_size_log2().is_some();
    for (b, &(start, len, first, count)) in plan.blocks.iter().enumerate() {
        let last = b + 1 == plan.blocks.len();
        let block_lens = &lens[first..first + count];
        let block_codes: usize = block_lens.iter().map(|&l| l as usize).sum();
        let index_len = if splits { 4 * count } else { 0 };
        let payload = index_len + block_codes;
        let chunk = &input[start..start + len];
        if payload >= len {
            out.extend_from_slice(&block_header(true, last, len, len));
            out.extend_from_slice(chunk);
        } else {
            out.extend_from_slice(&block_header(false, last, payload, len));
            if splits {
                let mut end = 0u32;
                for &l in block_lens {
                    end += l;
                    out.extend_from_slice(&end.to_le_bytes());
                }
            }
            out.extend_from_slice(&codes[code_pos..code_pos + block_codes]);
        }
        code_pos += block_codes;
    }
    if options.checksum {
        out.extend_from_slice(&xxh3_64(input).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_size_and_contents() {
        let dict = Dictionary::from_symbols(Scheme::U8, [&b"abc"[..], b"ab", b"x", b"hello wo"]).unwrap();
        let t = encode_table_bytes(&dict);
        assert_eq!(t.len(), ENCODE_TABLE_BYTES);
        let byte_code = &t[t.len() - 256..];
        assert_eq!(byte_code[usize::from(b'x')], 2);
        assert_eq!(byte_code[usize::from(b'y')], 255);
    }

    #[test]
    fn plan_covers_input() {
        for split in [None, Some(8), Some(12)] {
            let opts = FrameOptions {
                block_size_log2: 12,
                split_size_log2: split,
                ..FrameOptions::default()
            };
            let plan = plan_frame(10_000, opts);
            let mut pos = 0u64;
            for &(start, len) in &plan.pieces {
                assert_eq!(start, pos);
                pos += u64::from(len);
            }
            assert_eq!(pos, 10_000);
            assert_eq!(plan.blocks.len(), 3);
        }
    }
}
