//! The frame format: a header plus independently decodable blocks.
//!
//! See `docs/FORMAT.md` for the normative description.

use alloc::vec::Vec;

use xxhash_rust::xxh3::xxh3_64;

use crate::util::{Reader, put_u64};
use crate::{Compressor, Decompressor, DictId, Error, Result, Scheme};

pub(crate) const FRAME_MAGIC: [u8; 4] = *b"PRST";
pub(crate) const FRAME_VERSION: u8 = 1;

pub(crate) const FLAG_CHECKSUM: u8 = 1 << 0;
pub(crate) const FLAG_CONTENT_SIZE: u8 = 1 << 1;
/// Reserved: dictionary stored inline in the frame.
#[expect(dead_code, reason = "reserved in the v1 format, implemented in a later milestone")]
pub(crate) const FLAG_DICT_INLINE: u8 = 1 << 2;
/// Reserved: several dictionaries, selected per block.
#[expect(dead_code, reason = "reserved in the v1 format, implemented in a later milestone")]
pub(crate) const FLAG_MULTI_DICT: u8 = 1 << 3;
const SUPPORTED_FLAGS: u8 = FLAG_CHECKSUM | FLAG_CONTENT_SIZE;

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
}

impl Default for FrameOptions {
    fn default() -> Self {
        Self {
            block_size_log2: DEFAULT_BLOCK_SIZE_LOG2,
            checksum: true,
            content_size: true,
        }
    }
}

impl FrameOptions {
    pub(crate) fn block_size(self) -> usize {
        1 << self.block_size_log2.clamp(MIN_BLOCK_SIZE_LOG2, MAX_BLOCK_SIZE_LOG2)
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
}

impl FrameHeader {
    /// Parses the header at the start of `data`. Returns the header and its
    /// length in bytes.
    pub fn parse(data: &[u8]) -> Result<(Self, usize)> {
        let mut r = Reader::new(data);
        let header = Self::read(&mut r)?;
        Ok((header, r.pos()))
    }

    pub(crate) fn read(r: &mut Reader<'_>) -> Result<Self> {
        if r.array::<4>()? != FRAME_MAGIC {
            return Err(Error::BadMagic);
        }
        let version = r.u8()?;
        if version != FRAME_VERSION {
            return Err(Error::UnsupportedVersion(version as u16));
        }
        let flags = r.u8()?;
        if flags & !SUPPORTED_FLAGS != 0 {
            return Err(Error::UnsupportedFlags(flags));
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
        Ok(Self {
            scheme,
            block_size_log2,
            dict_id,
            checksum: flags & FLAG_CHECKSUM != 0,
            content_size,
        })
    }

    pub(crate) fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&FRAME_MAGIC);
        out.push(FRAME_VERSION);
        let mut flags = 0;
        if self.checksum {
            flags |= FLAG_CHECKSUM;
        }
        if self.content_size.is_some() {
            flags |= FLAG_CONTENT_SIZE;
        }
        out.push(flags);
        out.push(self.scheme.as_u8());
        out.push(self.block_size_log2);
        put_u64(out, self.dict_id.0);
        if let Some(size) = self.content_size {
            put_u64(out, size);
        }
    }

    /// Block size in bytes.
    #[must_use]
    pub fn block_size(&self) -> usize {
        1 << self.block_size_log2
    }
}

/// Appends one block (header + payload) for `chunk` to `out`. Falls back to a
/// raw block when coding does not save space.
pub(crate) fn write_block(comp: &Compressor, chunk: &[u8], last: bool, out: &mut Vec<u8>) {
    let header_pos = out.len();
    out.extend_from_slice(&[0; BLOCK_HEADER_LEN]);
    let csize = comp.compress_block_into(chunk, out);
    let kind = if csize >= chunk.len() {
        out.truncate(header_pos + BLOCK_HEADER_LEN);
        out.extend_from_slice(chunk);
        BLOCK_RAW
    } else {
        BLOCK_CODED
    };
    let csize = out.len() - header_pos - BLOCK_HEADER_LEN;
    let word = csize as u32 | if last { BLOCK_LAST } else { 0 } | (kind << BLOCK_TYPE_SHIFT);
    out[header_pos..header_pos + 4].copy_from_slice(&word.to_le_bytes());
    out[header_pos + 4..header_pos + 8].copy_from_slice(&(chunk.len() as u32).to_le_bytes());
}

pub(crate) fn compress_frame(comp: &Compressor, input: &[u8], options: FrameOptions, out: &mut Vec<u8>) {
    let block_size = options.block_size();
    let header = FrameHeader {
        scheme: comp.scheme(),
        block_size_log2: block_size.trailing_zeros() as u8,
        dict_id: comp.dict_id(),
        checksum: options.checksum,
        content_size: options.content_size.then_some(input.len() as u64),
    };
    out.reserve(24 + input.len() + input.len() / 2 + 16);
    header.write(out);
    if input.is_empty() {
        write_block(comp, &[], true, out);
    } else {
        let blocks = input.len().div_ceil(block_size);
        for (i, chunk) in input.chunks(block_size).enumerate() {
            write_block(comp, chunk, i + 1 == blocks, out);
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
    pub(crate) fn parse(bytes: [u8; BLOCK_HEADER_LEN], frame: &FrameHeader, max_expansion: usize) -> Result<Self> {
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

pub(crate) fn decompress_frame(dec: &Decompressor, frame: &[u8], out: &mut Vec<u8>) -> Result<usize> {
    let mut r = Reader::new(frame);
    let header = FrameHeader::read(&mut r)?;
    if header.scheme != dec.scheme() {
        return Err(Error::Corrupt("frame scheme does not match dictionary"));
    }
    dec.check_dict(header.dict_id)?;
    if let Some(size) = header.content_size {
        if size > (frame.len() as u64).saturating_mul(dec.max_expansion() as u64) {
            return Err(Error::Corrupt("content size is impossible"));
        }
        out.reserve(size as usize);
    }
    let start = out.len();
    loop {
        let bh = BlockHeader::parse(r.array()?, &header, dec.max_expansion())?;
        let payload = r.bytes(bh.csize)?;
        if bh.raw {
            out.extend_from_slice(payload);
        } else {
            dec.decompress_block_into(payload, bh.dsize, out)?;
        }
        if bh.last {
            break;
        }
    }
    let produced = out.len() - start;
    if let Some(size) = header.content_size {
        if size != produced as u64 {
            return Err(Error::Corrupt("content size does not match"));
        }
    }
    if header.checksum {
        let expected = r.u64()?;
        if expected != xxh3_64(&out[start..]) {
            return Err(Error::ChecksumMismatch);
        }
    }
    if r.remaining() != 0 {
        return Err(Error::Corrupt("trailing bytes after frame"));
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
            },
            FrameOptions {
                block_size_log2: 10,
                checksum: false,
                content_size: false,
            },
            FrameOptions::default(),
        ] {
            let frame = c.compress_with(&input, opts);
            assert!(frame.len() < input.len());
            assert_eq!(d.decompress(&frame).unwrap(), input);
            let (h, _) = FrameHeader::parse(&frame).unwrap();
            assert_eq!(h.dict_id, c.dict_id());
        }
    }

    #[test]
    fn empty_and_incompressible() {
        let (c, d) = codec();
        assert_eq!(d.decompress(&c.compress(b"")).unwrap(), b"");
        let noise: Vec<u8> = (0..3000u32)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 24) as u8)
            .collect();
        let frame = c.compress(&noise);
        assert!(frame.len() <= noise.len() + 48, "raw fallback keeps overhead small");
        assert_eq!(d.decompress(&frame).unwrap(), noise);
    }

    #[test]
    fn rejects_corruption() {
        let (c, d) = codec();
        let frame = c.compress(b"lorem ipsum dolor lorem ipsum dolor");
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
