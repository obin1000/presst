//! Turning a validated frame into independent units of GPU work.

use presst::FrameIndex;

/// Raw blocks are copied in pieces of this size, so their work spreads over
/// many threads like coded splits do.
const RAW_PIECE: usize = 4096;

/// One unit of GPU work: the `Split` struct of `kernels/presst_decode.cuh`.
///
/// Decode (or, if `raw`, copy) input bytes `in_start..in_start + in_len` of the
/// frame into exactly `out_len` bytes at `out_start` of the content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuSplit {
    /// Offset of the codes in the frame.
    pub in_start: u64,
    /// Offset of the output in the content (a multiple of 8).
    pub out_start: u64,
    /// Length of the codes.
    pub in_len: u32,
    /// Exact decoded length.
    pub out_len: u32,
    /// Copy instead of decode.
    pub raw: bool,
}

/// Size of the kernel's `Split` struct.
pub(crate) const SPLIT_BYTES: usize = 32;

/// The independent pieces of work of a frame, in output order.
///
/// Frames with a split index (format version 2) yield one piece per split.
/// Version 1 frames yield one piece per coded block, which decodes correctly
/// but leaves most of the GPU idle.
#[must_use]
pub fn frame_splits(index: &FrameIndex) -> Vec<GpuSplit> {
    let split_size = index.header.split_size();
    let mut out = Vec::new();
    for b in &index.blocks {
        if b.raw {
            let mut pos = 0;
            while pos < b.codes.len() {
                let n = RAW_PIECE.min(b.codes.len() - pos);
                out.push(GpuSplit {
                    in_start: (b.codes.start + pos) as u64,
                    out_start: (b.decoded.start + pos) as u64,
                    in_len: n as u32,
                    out_len: n as u32,
                    raw: true,
                });
                pos += n;
            }
            continue;
        }
        match split_size {
            Some(split) => {
                let mut prev = 0usize;
                for (i, &end) in b.split_ends.iter().enumerate() {
                    let start = i * split;
                    out.push(GpuSplit {
                        in_start: (b.codes.start + prev) as u64,
                        out_start: (b.decoded.start + start) as u64,
                        in_len: (end as usize - prev) as u32,
                        out_len: split.min(b.decoded.len() - start) as u32,
                        raw: false,
                    });
                    prev = end as usize;
                }
            }
            None => {
                if !b.decoded.is_empty() || !b.codes.is_empty() {
                    out.push(GpuSplit {
                        in_start: b.codes.start as u64,
                        out_start: b.decoded.start as u64,
                        in_len: b.codes.len() as u32,
                        out_len: b.decoded.len() as u32,
                        raw: false,
                    });
                }
            }
        }
    }
    out
}

/// Strings per group in the string kernels (one warp).
const GROUP: usize = 1 << presst::STRING_GROUP_SIZE_LOG2;

/// The decoded start of every group of 32 strings plus the total, computed from
/// the code streams and the symbol lengths (for batches without a decoded
/// index). Rejects codes that are not in the dictionary and escapes at the end
/// of a string, like the CPU decoder.
pub(crate) fn group_starts(batch: &presst::StringBatch, lens: &[u8; 256]) -> Result<Vec<u64>, presst::Error> {
    let n = batch.len();
    let mut starts = Vec::with_capacity(n.div_ceil(GROUP) + 1);
    let mut total = 0u64;
    for i in 0..n {
        if i % GROUP == 0 {
            starts.push(total);
        }
        let codes = batch
            .codes(i)
            .ok_or(presst::Error::Corrupt("string index out of range"))?;
        let mut k = 0;
        while k < codes.len() {
            let c = codes[k];
            if c == 255 {
                if k + 1 == codes.len() {
                    return Err(presst::Error::Corrupt("escape at the end of a string"));
                }
                total += 1;
                k += 2;
            } else {
                let l = lens[c as usize];
                if l == 0 {
                    return Err(presst::Error::Corrupt("code not in the dictionary"));
                }
                total += u64::from(l);
                k += 1;
            }
        }
    }
    starts.push(total);
    Ok(starts)
}

/// One unit of work per group of 32 strings: its codes and decoded range.
pub(crate) fn string_groups(batch: &presst::StringBatch, starts: &[u64]) -> Vec<GpuSplit> {
    let offsets = batch.offsets();
    let n = batch.len();
    (0..n.div_ceil(GROUP))
        .map(|g| {
            let first = g * GROUP;
            let last = (first + GROUP).min(n);
            GpuSplit {
                in_start: u64::from(offsets[first]),
                out_start: starts[g],
                in_len: offsets[last] - offsets[first],
                out_len: (starts[g + 1] - starts[g]) as u32,
                raw: false,
            }
        })
        .collect()
}

/// Serializes splits in the kernel's layout (`u64, u64, u32, u32, u32, u32`).
pub(crate) fn to_bytes(splits: &[GpuSplit]) -> Vec<u8> {
    let mut out = Vec::with_capacity(splits.len() * SPLIT_BYTES);
    for s in splits {
        out.extend_from_slice(&s.in_start.to_le_bytes());
        out.extend_from_slice(&s.out_start.to_le_bytes());
        out.extend_from_slice(&s.in_len.to_le_bytes());
        out.extend_from_slice(&s.out_len.to_le_bytes());
        out.extend_from_slice(&u32::from(s.raw).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use presst::{Compressor, Dictionary, FrameOptions, Scheme};

    #[test]
    fn splits_cover_the_content() {
        let dict = Dictionary::from_symbols(Scheme::U8, [&b"lorem "[..], b"ipsum "]).unwrap();
        let input: Vec<u8> = b"lorem ipsum dolor ".iter().copied().cycle().take(10_000).collect();
        for split in [None, Some(8), Some(12)] {
            let frame = Compressor::new(&dict).unwrap().compress_with(
                &input,
                FrameOptions {
                    block_size_log2: 12,
                    split_size_log2: split,
                    ..FrameOptions::default()
                },
            );
            let index = FrameIndex::parse(&frame).unwrap();
            let splits = frame_splits(&index);
            let mut pos = 0u64;
            for s in &splits {
                assert_eq!(s.out_start, pos);
                assert_eq!(s.out_start % 8, 0);
                assert!(s.in_start + u64::from(s.in_len) <= frame.len() as u64);
                pos += u64::from(s.out_len);
            }
            assert_eq!(pos, input.len() as u64);
            assert_eq!(to_bytes(&splits).len(), splits.len() * SPLIT_BYTES);
        }
    }
}
