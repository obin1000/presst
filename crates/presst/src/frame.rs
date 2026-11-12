//! The frame format: a header plus independently decodable blocks.
//!
//! See `docs/FORMAT.md` for the normative description.

use alloc::vec::Vec;
use core::ops::Range;

use xxhash_rust::xxh3::xxh3_64;

use crate::util::{Reader, put_u64};
use crate::{Compressor, Decompressor, DictId, Error, Result, Scheme};

pub(crate) const FRAME_MAGIC: [u8; 4] = *b"PRST";
/// Version written for frames without a split index.
pub(crate) const FRAME_VERSION: u8 = 1;
/// Version written for frames with a split index (flag `SPLIT_INDEX`).
pub(crate) const FRAME_VERSION_SPLIT: u8 = 2;

pub(crate) const FLAG_CHECKSUM: u8 = 1 << 0;
pub(crate) const FLAG_CONTENT_SIZE: u8 = 1 << 1;
/// Reserved: dictionary stored inline in the frame.
#[expect(dead_code, reason = "reserved in the format, implemented in a later milestone")]
pub(crate) const FLAG_DICT_INLINE: u8 = 1 << 2;
/// Reserved: several dictionaries, selected per block.
#[expect(dead_code, reason = "reserved in the format, implemented in a later milestone")]
pub(crate) const FLAG_MULTI_DICT: u8 = 1 << 3;
/// Coded blocks start with a split index (format version 2).
pub(crate) const FLAG_SPLIT_INDEX: u8 = 1 << 4;
const SUPPORTED_FLAGS: u8 = FLAG_CHECKSUM | FLAG_CONTENT_SIZE | FLAG_SPLIT_INDEX;

pub(crate) const BLOCK_HEADER_LEN: usize = 8;
const BLOCK_CSIZE_MASK: u32 = (1 << 29) - 1;
const BLOCK_LAST: u32 = 1 << 29;
const BLOCK_TYPE_SHIFT: u32 = 30;
const BLOCK_CODED: u32 = 0;
const BLOCK_RAW: u32 = 1;

/// Smallest allowed block size, as a power of two (1 KiB).
pub const MIN_BLOCK_SIZE_LOG2: u8 = 10;
/// Largest allowed block size, as a power of two (64 MiB).
pub const MAX_BLOCK_SIZE_LOG2: u8 = 26;
/// Default block size, as a power of two (1 MiB).
pub const DEFAULT_BLOCK_SIZE_LOG2: u8 = 20;
/// Smallest allowed split size, as a power of two (256 bytes).
pub const MIN_SPLIT_SIZE_LOG2: u8 = 8;
/// Split size used by GPU-oriented writers, as a power of two (4 KiB).
pub const DEFAULT_SPLIT_SIZE_LOG2: u8 = 12;

/// Options for writing frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameOptions {
    /// Block size as a power of two, between [`MIN_BLOCK_SIZE_LOG2`] and
    /// [`MAX_BLOCK_SIZE_LOG2`]. Values outside the range are clamped.
    ///
    /// Blocks exist for parallelism and to store incompressible data raw;
    /// unlike FSST there is no per-block table to amortize, so the default is
    /// large.
    pub block_size_log2: u8,
    /// Append an xxh3-64 checksum of the decompressed content.
    pub checksum: bool,
    /// Store the decompressed size in the header. Ignored by the streaming
    /// writer, which does not know it up front.
    pub content_size: bool,
    /// Cut every coded block into splits of `2^split_size_log2` input bytes
    /// that are encoded independently, and store where each split's codes
    /// start (format version 2). Every split then decodes on its own at a
    /// known output position, which GPU and multi-threaded decoders need.
    /// Costs 4 bytes per split plus a little ratio at split boundaries.
    ///
    /// Clamped to [`MIN_SPLIT_SIZE_LOG2`] and the block size. `None` (the
    /// default) writes version 1 frames without an index.
    pub split_size_log2: Option<u8>,
}

impl Default for FrameOptions {
    fn default() -> Self {
        Self {
            block_size_log2: DEFAULT_BLOCK_SIZE_LOG2,
            checksum: true,
            content_size: true,
            split_size_log2: None,
        }
    }
}

impl FrameOptions {
    /// The block size that is written, as a power of two (the field, clamped
    /// to the allowed range).
    #[must_use]
    pub fn block_size_log2(self) -> u8 {
        self.block_size_log2.clamp(MIN_BLOCK_SIZE_LOG2, MAX_BLOCK_SIZE_LOG2)
    }

    /// The block size that is written, in bytes.
    #[must_use]
    pub fn block_size(self) -> usize {
        1 << self.block_size_log2()
    }

    /// The split size that is written, as a power of two (the field, clamped
    /// to the allowed range), if any.
    #[must_use]
    pub fn split_size_log2(self) -> Option<u8> {
        self.split_size_log2
            .map(|s| s.clamp(MIN_SPLIT_SIZE_LOG2, self.block_size_log2()))
    }
}

/// Parsed frame header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    /// Code scheme of the referenced dictionary.
    pub scheme: Scheme,
    /// Block size as a power of two.
    pub block_size_log2: u8,
    /// Dictionary needed to decode the frame.
    pub dict_id: DictId,
    /// Whether a content checksum follows the last block.
    pub checksum: bool,
    /// Decompressed size, if stored.
    pub content_size: Option<u64>,
    /// Split size as a power of two, if coded blocks carry a split index.
    pub split_size_log2: Option<u8>,
}

impl FrameHeader {
    /// Parses the header at the start of `data`. Returns the header and its
    /// length in bytes.
    pub fn parse(data: &[u8]) -> Result<(Self, usize)> {
        let mut r = Reader::new(data);
        let header = Self::read(&mut r)?;
        Ok((header, r.pos()))
    }

    /// Length of the header that starts with these first 16 bytes.
    #[cfg(feature = "std")]
    pub(crate) fn len_from_prefix(prefix: &[u8; 16]) -> usize {
        let flags = prefix[5];
        16 + if flags & FLAG_CONTENT_SIZE != 0 { 8 } else { 0 } + usize::from(flags & FLAG_SPLIT_INDEX != 0)
    }

    pub(crate) fn read(r: &mut Reader<'_>) -> Result<Self> {
        if r.array::<4>()? != FRAME_MAGIC {
            return Err(Error::BadMagic);
        }
        let version = r.u8()?;
        if version != FRAME_VERSION && version != FRAME_VERSION_SPLIT {
            return Err(Error::UnsupportedVersion(version as u16));
        }
        let flags = r.u8()?;
        if flags & !SUPPORTED_FLAGS != 0 {
            return Err(Error::UnsupportedFlags(flags));
        }
        // The split index is the only version 2 feature, and it needs version 2.
        if (flags & FLAG_SPLIT_INDEX != 0) != (version == FRAME_VERSION_SPLIT) {
            return Err(Error::Corrupt("split index flag does not match the version"));
        }
        let scheme = Scheme::from_u8(r.u8()?)?;
        let block_size_log2 = r.u8()?;
        if !(MIN_BLOCK_SIZE_LOG2..=MAX_BLOCK_SIZE_LOG2).contains(&block_size_log2) {
            return Err(Error::Corrupt("block size out of range"));
        }
        let dict_id = DictId(r.u64()?);
        let content_size = if flags & FLAG_CONTENT_SIZE != 0 {
            Some(r.u64()?)
        } else {
            None
        };
        let split_size_log2 = if flags & FLAG_SPLIT_INDEX != 0 {
            let s = r.u8()?;
            if !(MIN_SPLIT_SIZE_LOG2..=block_size_log2).contains(&s) {
                return Err(Error::Corrupt("split size out of range"));
            }
            Some(s)
        } else {
            None
        };
        Ok(Self {
            scheme,
            block_size_log2,
            dict_id,
            checksum: flags & FLAG_CHECKSUM != 0,
            content_size,
            split_size_log2,
        })
    }

    /// Appends the serialized header to `out`. For writers that assemble
    /// frames themselves (for example on a GPU); see [`block_header`].
    pub fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&FRAME_MAGIC);
        out.push(if self.split_size_log2.is_some() {
            FRAME_VERSION_SPLIT
        } else {
            FRAME_VERSION
        });
        let mut flags = 0;
        if self.checksum {
            flags |= FLAG_CHECKSUM;
        }
        if self.content_size.is_some() {
            flags |= FLAG_CONTENT_SIZE;
        }
        if self.split_size_log2.is_some() {
            flags |= FLAG_SPLIT_INDEX;
        }
        out.push(flags);
        out.push(self.scheme.as_u8());
        out.push(self.block_size_log2);
        put_u64(out, self.dict_id.0);
        if let Some(size) = self.content_size {
            put_u64(out, size);
        }
        if let Some(s) = self.split_size_log2 {
            out.push(s);
        }
    }

    /// Block size in bytes.
    #[must_use]
    pub fn block_size(&self) -> usize {
        1 << self.block_size_log2
    }

    /// Split size in bytes, if coded blocks carry a split index.
    #[must_use]
    pub fn split_size(&self) -> Option<usize> {
        self.split_size_log2.map(|s| 1 << s)
    }
}

/// The 8-byte header of a block whose payload is `csize` bytes and decodes to
/// `dsize` bytes. For writers that assemble frames themselves: such a writer
/// must follow the rules in `docs/FORMAT.md` (block sizes, raw fallback, split
/// index) to produce frames identical to [`Compressor::compress_with`].
#[must_use]
pub fn block_header(raw: bool, last: bool, csize: usize, dsize: usize) -> [u8; BLOCK_HEADER_LEN] {
    let kind = if raw { BLOCK_RAW } else { BLOCK_CODED };
    let word = csize as u32 | if last { BLOCK_LAST } else { 0 } | (kind << BLOCK_TYPE_SHIFT);
    let mut out = [0u8; BLOCK_HEADER_LEN];
    out[..4].copy_from_slice(&word.to_le_bytes());
    out[4..].copy_from_slice(&(dsize as u32).to_le_bytes());
    out
}

/// Appends one block (header + payload) for `chunk` to `out`. Falls back to a
/// raw block when coding does not save space.
pub(crate) fn write_block(comp: &Compressor, chunk: &[u8], last: bool, split_size: Option<usize>, out: &mut Vec<u8>) {
    let header_pos = out.len();
    out.extend_from_slice(&[0; BLOCK_HEADER_LEN]);
    let payload_pos = out.len();
    match split_size {
        None => {
            comp.compress_block_into(chunk, out);
        }
        Some(split) => {
            // Index of code end offsets, then the codes of each split, each
            // encoded on its own.
            let splits = chunk.len().div_ceil(split);
            out.resize(payload_pos + 4 * splits, 0);
            let codes_pos = out.len();
            for (i, part) in chunk.chunks(split).enumerate() {
                comp.compress_block_into(part, out);
                let end = (out.len() - codes_pos) as u32;
                out[payload_pos + 4 * i..payload_pos + 4 * i + 4].copy_from_slice(&end.to_le_bytes());
            }
        }
    }
    let raw = out.len() - payload_pos >= chunk.len();
    if raw {
        out.truncate(payload_pos);
        out.extend_from_slice(chunk);
    }
    let csize = out.len() - payload_pos;
    out[header_pos..header_pos + BLOCK_HEADER_LEN].copy_from_slice(&block_header(raw, last, csize, chunk.len()));
}

pub(crate) fn compress_frame(comp: &Compressor, input: &[u8], options: FrameOptions, out: &mut Vec<u8>) {
    let block_size = options.block_size();
    let split_size_log2 = options.split_size_log2();
    let split_size = split_size_log2.map(|s| 1usize << s);
    let header = FrameHeader {
        scheme: comp.scheme(),
        block_size_log2: options.block_size_log2(),
        dict_id: comp.dict_id(),
        checksum: options.checksum,
        content_size: options.content_size.then_some(input.len() as u64),
        split_size_log2,
    };
    out.reserve(25 + input.len() + input.len() / 2 + 16);
    header.write(out);
    let blocks = input.len().div_ceil(block_size);
    if input.is_empty() {
        write_block(comp, &[], true, split_size, out);
    } else if cfg!(feature = "parallel") && blocks > 1 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let parts: Vec<Vec<u8>> = input
                .par_chunks(block_size)
                .enumerate()
                .map(|(i, chunk)| {
                    let mut part = Vec::with_capacity(chunk.len() + BLOCK_HEADER_LEN + 16);
                    write_block(comp, chunk, i + 1 == blocks, split_size, &mut part);
                    part
                })
                .collect();
            for part in parts {
                out.extend_from_slice(&part);
            }
        }
    } else {
        for (i, chunk) in input.chunks(block_size).enumerate() {
            write_block(comp, chunk, i + 1 == blocks, split_size, out);
        }
    }
    if options.checksum {
        put_u64(out, xxh3_64(input));
    }
}

/// A parsed block header.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlockHeader {
    pub(crate) raw: bool,
    pub(crate) last: bool,
    pub(crate) csize: usize,
    pub(crate) dsize: usize,
}

impl BlockHeader {
    pub(crate) fn parse(bytes: [u8; BLOCK_HEADER_LEN], frame: &FrameHeader) -> Result<Self> {
        let max_expansion = frame.scheme.max_symbol_len();
        let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let dsize = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let csize = (word & BLOCK_CSIZE_MASK) as usize;
        let last = word & BLOCK_LAST != 0;
        let raw = match word >> BLOCK_TYPE_SHIFT {
            BLOCK_CODED => false,
            BLOCK_RAW => true,
            _ => return Err(Error::Corrupt("unknown block type")),
        };
        if dsize > frame.block_size() {
            return Err(Error::Corrupt("block larger than the frame's block size"));
        }
        if raw && csize != dsize {
            return Err(Error::Corrupt("raw block sizes differ"));
        }
        // Bounds allocations by the input size: a code expands to at most
        // `max_expansion` bytes.
        if !raw && dsize > csize.saturating_mul(max_expansion) {
            return Err(Error::Corrupt("block decoded size is impossible"));
        }
        Ok(Self {
            raw,
            last,
            csize,
            dsize,
        })
    }
}

/// Where one block of a frame is, and how it decodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockInfo {
    /// Stored raw: `codes` is the decoded data itself.
    pub raw: bool,
    /// Byte range of the block's code stream (or raw data) in the frame,
    /// without the block header and split index.
    pub codes: Range<usize>,
    /// Byte range of the block's decoded data in the decompressed content.
    pub decoded: Range<usize>,
    /// For coded blocks of frames with a split index: the end of each split's
    /// codes, relative to `codes.start`. Split `i` covers
    /// `codes.start + split_ends[i-1] .. codes.start + split_ends[i]` (0 for
    /// the first) and decodes to `split_size` bytes (the last split: the
    /// rest of the block) at `decoded.start + i * split_size`.
    pub split_ends: Vec<u32>,
}

/// A validated index of a frame: its header and where every block is.
///
/// Built without decoding anything, so other decoders (multi-threaded, GPU)
/// can schedule blocks and splits independently. Every size and offset is
/// checked against the frame length and the format's limits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameIndex {
    /// The frame header.
    pub header: FrameHeader,
    /// Blocks in order.
    pub blocks: Vec<BlockInfo>,
    /// Total decompressed size.
    pub content_size: usize,
    /// The stored content checksum, if any.
    pub checksum: Option<u64>,
}

impl FrameIndex {
    /// Parses and validates `frame`. Fails on any structural problem; the code
    /// streams themselves are only checked when decoding.
    pub fn parse(frame: &[u8]) -> Result<Self> {
        let mut r = Reader::new(frame);
        let header = FrameHeader::read(&mut r)?;
        if let Some(size) = header.content_size {
            if size > (frame.len() as u64).saturating_mul(header.scheme.max_symbol_len() as u64) {
                return Err(Error::Corrupt("content size is impossible"));
            }
        }
        let mut blocks = Vec::new();
        let mut decoded = 0usize;
        loop {
            let bh = BlockHeader::parse(r.array()?, &header)?;
            let start = r.pos();
            let payload = r.bytes(bh.csize)?;
            blocks.push(block_info(&header, &bh, payload, start, decoded)?);
            decoded += bh.dsize;
            if bh.last {
                break;
            }
        }
        if let Some(size) = header.content_size {
            if size != decoded as u64 {
                return Err(Error::Corrupt("content size does not match"));
            }
        }
        let checksum = if header.checksum { Some(r.u64()?) } else { None };
        if r.remaining() != 0 {
            return Err(Error::Corrupt("trailing bytes after frame"));
        }
        Ok(Self {
            header,
            blocks,
            content_size: decoded,
            checksum,
        })
    }
}

/// Validates one block's payload (as stored after its header, at offset
/// `payload_start` of the frame) and describes it. For frames with a split
/// index, this checks that the index is ordered, covers the codes exactly and
/// that every split can decode to its size.
pub(crate) fn block_info(
    header: &FrameHeader,
    bh: &BlockHeader,
    payload: &[u8],
    payload_start: usize,
    decoded_start: usize,
) -> Result<BlockInfo> {
    let mut codes = payload_start..payload_start + payload.len();
    let mut split_ends = Vec::new();
    if let (false, Some(split)) = (bh.raw, header.split_size()) {
        let splits = bh.dsize.div_ceil(split);
        let index_len = 4 * splits;
        if index_len > payload.len() {
            return Err(Error::Corrupt("split index larger than the block"));
        }
        let codes_len = payload.len() - index_len;
        let mut prev = 0usize;
        for i in 0..splits {
            let b = &payload[4 * i..4 * i + 4];
            let end = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            let e = end as usize;
            let split_dsize = split.min(bh.dsize - i * split);
            if e < prev || e > codes_len {
                return Err(Error::Corrupt("split index out of order"));
            }
            if split_dsize > (e - prev).saturating_mul(header.scheme.max_symbol_len()) {
                return Err(Error::Corrupt("split decoded size is impossible"));
            }
            split_ends.push(end);
            prev = e;
        }
        if prev != codes_len {
            return Err(Error::Corrupt("split index does not cover the block"));
        }
        codes.start += index_len;
    }
    Ok(BlockInfo {
        raw: bh.raw,
        codes,
        decoded: decoded_start..decoded_start + bh.dsize,
        split_ends,
    })
}

/// Decodes one block's payload (as stored after its header) and appends the
/// result to `out`.
pub(crate) fn decode_block(
    dec: &Decompressor,
    frame: &[u8],
    block: &BlockInfo,
    split_size: Option<usize>,
    out: &mut Vec<u8>,
) -> Result<()> {
    let codes = &frame[block.codes.clone()];
    if block.raw {
        out.extend_from_slice(codes);
        return Ok(());
    }
    match split_size {
        None => dec.decompress_block_into(codes, block.decoded.len(), out),
        Some(split) => {
            let mut prev = 0usize;
            for (i, &end) in block.split_ends.iter().enumerate() {
                let dsize = split.min(block.decoded.len() - i * split);
                dec.decompress_block_into(&codes[prev..end as usize], dsize, out)?;
                prev = end as usize;
            }
            Ok(())
        }
    }
}

pub(crate) fn decompress_frame(dec: &Decompressor, frame: &[u8], out: &mut Vec<u8>) -> Result<usize> {
    // Parse and validate the whole block index first: it checks sizes before
    // any decoding and lets blocks be decoded in parallel.
    let index = FrameIndex::parse(frame)?;
    let header = index.header;
    if header.scheme != dec.scheme() {
        return Err(Error::Corrupt("frame scheme does not match dictionary"));
    }
    dec.check_dict(header.dict_id)?;
    let split = header.split_size();
    out.reserve(index.content_size);
    let start = out.len();
    if cfg!(feature = "parallel") && index.blocks.len() > 1 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let parts: Vec<Vec<u8>> = index
                .blocks
                .par_iter()
                .map(|b| {
                    let mut part = Vec::with_capacity(b.decoded.len());
                    decode_block(dec, frame, b, split, &mut part).map(|()| part)
                })
                .collect::<Result<_>>()?;
            for part in parts {
                out.extend_from_slice(&part);
            }
        }
    } else {
        for b in &index.blocks {
            decode_block(dec, frame, b, split, out)?;
        }
    }
    let produced = out.len() - start;
    if produced != index.content_size {
        return Err(Error::Corrupt("decoded size does not match"));
    }
    if let Some(expected) = index.checksum {
        if expected != xxh3_64(&out[start..]) {
            return Err(Error::ChecksumMismatch);
        }
    }
    Ok(produced)
}

/// Incremental checksum state used by the streaming reader and writer.
#[cfg(feature = "std")]
pub(crate) struct ContentHasher(xxhash_rust::xxh3::Xxh3);

#[cfg(feature = "std")]
impl ContentHasher {
    pub(crate) fn new() -> Self {
        Self(xxhash_rust::xxh3::Xxh3::new())
    }

    pub(crate) fn update(&mut self, data: &[u8]) {
        self.0.update(data);
    }

    pub(crate) fn finish(&self) -> u64 {
        self.0.digest()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dictionary;
    use alloc::vec;

    fn codec() -> (Compressor, Decompressor) {
        let dict = Dictionary::from_symbols(Scheme::U8, [&b"lorem "[..], b"ipsum ", b"dolor", b" ", b"e"]).unwrap();
        (Compressor::new(&dict).unwrap(), Decompressor::new(&dict).unwrap())
    }

    fn opts(block_size_log2: u8, split_size_log2: Option<u8>) -> FrameOptions {
        FrameOptions {
            block_size_log2,
            split_size_log2,
            ..FrameOptions::default()
        }
    }

    #[test]
    fn roundtrip_multi_block() {
        let (c, d) = codec();
        let input: Vec<u8> = b"lorem ipsum dolor sit amet "
            .iter()
            .copied()
            .cycle()
            .take(5000)
            .collect();
        for opts in [
            FrameOptions {
                block_size_log2: 10,
                checksum: true,
                content_size: true,
                split_size_log2: None,
            },
            FrameOptions {
                block_size_log2: 10,
                checksum: false,
                content_size: false,
                split_size_log2: None,
            },
            FrameOptions::default(),
            opts(10, Some(8)),
            opts(12, Some(9)),
            FrameOptions {
                checksum: false,
                content_size: false,
                ..opts(20, Some(DEFAULT_SPLIT_SIZE_LOG2))
            },
        ] {
            let frame = c.compress_with(&input, opts);
            assert!(frame.len() < input.len());
            assert_eq!(d.decompress(&frame).unwrap(), input);
            let (h, _) = FrameHeader::parse(&frame).unwrap();
            assert_eq!(h.dict_id, c.dict_id());
            assert_eq!(h.split_size_log2, opts.split_size_log2());
        }
    }

    #[test]
    fn split_index_layout() {
        let (c, d) = codec();
        let input: Vec<u8> = b"lorem ipsum dolor sit amet "
            .iter()
            .copied()
            .cycle()
            .take(3000)
            .collect();
        let frame = c.compress_with(&input, opts(10, Some(8)));
        assert_eq!(frame[4], FRAME_VERSION_SPLIT);
        let index = FrameIndex::parse(&frame).unwrap();
        assert_eq!(index.content_size, 3000);
        assert_eq!(index.blocks.len(), 3);
        for b in &index.blocks {
            assert!(!b.raw);
            assert_eq!(b.split_ends.len(), b.decoded.len().div_ceil(256));
            assert_eq!(*b.split_ends.last().unwrap() as usize, b.codes.len());
            // Every split decodes on its own to its slice of the input.
            let mut prev = 0usize;
            for (i, &end) in b.split_ends.iter().enumerate() {
                let mut out = Vec::new();
                let start = b.decoded.start + i * 256;
                let len = 256.min(b.decoded.end - start);
                d.decompress_block_into(
                    &frame[b.codes.start + prev..b.codes.start + end as usize],
                    len,
                    &mut out,
                )
                .unwrap();
                assert_eq!(out, input[start..start + len]);
                prev = end as usize;
            }
        }
        // Splits restart the encoder, which costs only a little ratio.
        let plain = c.compress_with(&input, opts(10, None));
        assert!(
            frame.len() < plain.len() + plain.len() / 10,
            "{} vs {}",
            frame.len(),
            plain.len()
        );
    }

    #[test]
    fn split_size_is_clamped() {
        assert_eq!(opts(10, Some(2)).split_size_log2(), Some(MIN_SPLIT_SIZE_LOG2));
        assert_eq!(opts(10, Some(20)).split_size_log2(), Some(10));
    }

    #[test]
    fn empty_and_incompressible() {
        let (c, d) = codec();
        for o in [FrameOptions::default(), opts(10, Some(8))] {
            assert_eq!(d.decompress(&c.compress_with(b"", o)).unwrap(), b"");
            let noise: Vec<u8> = (0..3000u32)
                .map(|i| (i.wrapping_mul(2_654_435_761) >> 24) as u8)
                .collect();
            let frame = c.compress_with(&noise, o);
            // Header (at most 25 bytes), 8 bytes per block, checksum.
            assert!(
                frame.len() <= noise.len() + 25 + 3 * 8 + 8,
                "raw fallback keeps overhead small"
            );
            assert_eq!(d.decompress(&frame).unwrap(), noise);
        }
    }

    #[test]
    fn rejects_corruption() {
        let (c, d) = codec();
        for o in [FrameOptions::default(), opts(10, Some(8))] {
            let frame = c.compress_with(b"lorem ipsum dolor lorem ipsum dolor lorem ipsum dolor sit amet", o);
            for i in 0..frame.len() {
                let mut bad = frame.clone();
                bad[i] ^= 0x41;
                assert!(d.decompress(&bad).is_err(), "flip at {i} not detected");
            }
            for n in 0..frame.len() {
                assert!(d.decompress(&frame[..n]).is_err());
            }
            let mut trailing = frame.clone();
            trailing.push(0);
            assert!(d.decompress(&trailing).is_err());
            let mut out = vec![1, 2, 3];
            assert!(d.decompress_into(&frame[..frame.len() - 1], &mut out).is_err());
            assert_eq!(out, [1, 2, 3], "output is restored on error");
        }
    }

    #[test]
    fn rejects_bad_split_index() {
        let (c, d) = codec();
        let input: Vec<u8> = b"lorem ipsum dolor ".iter().copied().cycle().take(1000).collect();
        let frame = c.compress_with(
            &input,
            FrameOptions {
                checksum: false,
                ..opts(10, Some(8))
            },
        );
        let index = FrameIndex::parse(&frame).unwrap();
        let first_entry = index.blocks[0].codes.start - 4 * index.blocks[0].split_ends.len();
        // Splits out of order.
        let mut bad = frame.clone();
        bad[first_entry..first_entry + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(FrameIndex::parse(&bad).is_err());
        assert!(d.decompress(&bad).is_err());
        // Version 2 without the flag, and the flag in version 1.
        let mut bad = frame.clone();
        bad[4] = FRAME_VERSION;
        assert!(FrameHeader::parse(&bad).is_err());
        let mut bad = c.compress_with(&input, opts(10, None));
        bad[4] = FRAME_VERSION_SPLIT;
        assert!(FrameHeader::parse(&bad).is_err());
    }

    #[test]
    fn wrong_dictionary() {
        let (c, _) = codec();
        let other = Dictionary::from_symbols(Scheme::U8, [&b"x"[..]]).unwrap();
        let d = Decompressor::new(&other).unwrap();
        assert!(matches!(
            d.decompress(&c.compress(b"abc")),
            Err(Error::DictionaryMismatch { .. })
        ));
    }
}
