//! Batches of individually compressed strings, for columnar use.

use alloc::vec::Vec;

use crate::util::{Reader, put_u32, put_u64};
use crate::{DictId, Error, Result};

const BATCH_MAGIC: [u8; 4] = *b"PRSC";
const BATCH_VERSION: u8 = 1;
/// Version of batches with a decoded index.
const BATCH_VERSION_INDEXED: u8 = 2;

/// Strings per group of the decoded index, as a power of two (32, one GPU warp).
pub const STRING_GROUP_SIZE_LOG2: u8 = 5;

/// Options for [`Compressor::compress_strings_with`](crate::Compressor::compress_strings_with).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StringOptions {
    /// Also store the decoded start of every group of
    /// `2^`[`STRING_GROUP_SIZE_LOG2`] strings (8 bytes per group, format
    /// version 2), so that GPU and multi-threaded decoders know where each
    /// group's output goes without decoding the strings before it.
    pub decoded_index: bool,
}

/// Many strings compressed with the same dictionary, each independently
/// decodable.
///
/// This is the layout database engines use for FSST-compressed string columns:
/// one buffer of code streams plus `len + 1` offsets into it. Create one with
/// [`Compressor::compress_strings`](crate::Compressor::compress_strings) and
/// read strings back with
/// [`Decompressor::decompress_string`](crate::Decompressor::decompress_string).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StringBatch {
    dict_id: DictId,
    codes: Vec<u8>,
    offsets: Vec<u32>,
    /// Group size of the decoded index, as a power of two (0 if none).
    group_size_log2: u8,
    /// Decoded start of strings `0, g, 2g, ...` plus the total decoded size;
    /// empty without a decoded index.
    group_starts: Vec<u64>,
}

impl StringBatch {
    pub(crate) fn from_parts(dict_id: DictId, codes: Vec<u8>, offsets: Vec<u32>) -> Self {
        debug_assert_eq!(offsets.first(), Some(&0));
        Self {
            dict_id,
            codes,
            offsets,
            group_size_log2: 0,
            group_starts: Vec::new(),
        }
    }

    /// Records the decoded index from the decoded length of every string.
    pub(crate) fn set_decoded_index(&mut self, decoded_lens: &[usize]) {
        let g = 1usize << STRING_GROUP_SIZE_LOG2;
        let mut starts = Vec::with_capacity(decoded_lens.len().div_ceil(g) + 1);
        let mut total = 0u64;
        for (i, &len) in decoded_lens.iter().enumerate() {
            if i % g == 0 {
                starts.push(total);
            }
            total += len as u64;
        }
        starts.push(total);
        self.group_size_log2 = STRING_GROUP_SIZE_LOG2;
        self.group_starts = starts;
    }

    /// Creates a batch from existing parts, validating the offsets.
    pub fn new(dict_id: DictId, codes: Vec<u8>, offsets: Vec<u32>) -> Result<Self> {
        if offsets.first() != Some(&0) {
            return Err(Error::Corrupt("first offset must be 0"));
        }
        if offsets.windows(2).any(|w| w[0] > w[1]) {
            return Err(Error::Corrupt("offsets are not sorted"));
        }
        if *offsets.last().unwrap_or(&0) as usize != codes.len() {
            return Err(Error::Corrupt("last offset does not match code length"));
        }
        Ok(Self {
            dict_id,
            codes,
            offsets,
            group_size_log2: 0,
            group_starts: Vec::new(),
        })
    }

    /// Adds a decoded index (see [`StringOptions::decoded_index`]): the decoded
    /// start of strings `0, 2^group_size_log2, ...` followed by the total
    /// decoded size. Validates its structure against the code offsets; whether
    /// the strings really decode to these sizes is checked when decoding.
    pub fn with_decoded_index(mut self, group_size_log2: u8, group_starts: Vec<u64>) -> Result<Self> {
        if !(1..=16).contains(&group_size_log2) {
            return Err(Error::Corrupt("string group size out of range"));
        }
        let g = 1usize << group_size_log2;
        let groups = self.len().div_ceil(g);
        if group_starts.len() != groups + 1 || group_starts.first() != Some(&0) {
            return Err(Error::Corrupt("decoded index has the wrong length"));
        }
        for (i, w) in group_starts.windows(2).enumerate() {
            let codes = u64::from(self.offsets[((i + 1) * g).min(self.len())] - self.offsets[i * g]);
            // A code decodes to at most 8 bytes.
            if w[1] < w[0] || w[1] - w[0] > codes.saturating_mul(8) {
                return Err(Error::Corrupt("decoded index is impossible"));
            }
        }
        self.group_size_log2 = group_size_log2;
        self.group_starts = group_starts;
        Ok(self)
    }

    /// The decoded index, if any: the group size as a power of two, and the
    /// decoded start of every group followed by the total decoded size.
    #[must_use]
    pub fn decoded_index(&self) -> Option<(u8, &[u64])> {
        (!self.group_starts.is_empty()).then_some((self.group_size_log2, &self.group_starts[..]))
    }

    /// Dictionary the strings were compressed with.
    #[must_use]
    pub fn dict_id(&self) -> DictId {
        self.dict_id
    }

    /// Number of strings.
    #[must_use]
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Whether the batch holds no strings.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The code stream of string `index`, or `None` if out of range.
    #[must_use]
    pub fn codes(&self, index: usize) -> Option<&[u8]> {
        let start = *self.offsets.get(index)? as usize;
        let end = *self.offsets.get(index + 1)? as usize;
        self.codes.get(start..end)
    }

    /// Total size of all code streams in bytes.
    #[must_use]
    pub fn codes_len(&self) -> usize {
        self.codes.len()
    }

    /// All code streams, concatenated.
    #[must_use]
    pub fn code_bytes(&self) -> &[u8] {
        &self.codes
    }

    /// The `len + 1` offsets into [`code_bytes`](Self::code_bytes).
    #[must_use]
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    /// Serializes the batch (`PRSC` container, see `docs/FORMAT.md`).
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(18 + self.offsets.len() * 4 + self.group_starts.len() * 8 + self.codes.len());
        out.extend_from_slice(&BATCH_MAGIC);
        let indexed = !self.group_starts.is_empty();
        out.push(if indexed { BATCH_VERSION_INDEXED } else { BATCH_VERSION });
        put_u64(&mut out, self.dict_id.0);
        put_u32(&mut out, self.len() as u32);
        if indexed {
            out.push(self.group_size_log2);
        }
        for &o in &self.offsets {
            put_u32(&mut out, o);
        }
        for &s in &self.group_starts {
            put_u64(&mut out, s);
        }
        out.extend_from_slice(&self.codes);
        out
    }

    /// Parses a serialized batch.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut r = Reader::new(data);
        if r.array::<4>()? != BATCH_MAGIC {
            return Err(Error::BadMagic);
        }
        let version = r.u8()?;
        if version != BATCH_VERSION && version != BATCH_VERSION_INDEXED {
            return Err(Error::UnsupportedVersion(version as u16));
        }
        let dict_id = DictId(r.u64()?);
        let count = r.u32()? as usize;
        let group_size_log2 = if version == BATCH_VERSION_INDEXED {
            let g = r.u8()?;
            if !(1..=16).contains(&g) {
                return Err(Error::Corrupt("string group size out of range"));
            }
            Some(g)
        } else {
            None
        };
        if count.saturating_add(1).saturating_mul(4) > r.remaining() {
            return Err(Error::Truncated);
        }
        let mut offsets = Vec::with_capacity(count + 1);
        for _ in 0..=count {
            offsets.push(r.u32()?);
        }
        let mut group_starts = Vec::new();
        if let Some(g) = group_size_log2 {
            let n = count.div_ceil(1 << g) + 1;
            if n.saturating_mul(8) > r.remaining() {
                return Err(Error::Truncated);
            }
            for _ in 0..n {
                group_starts.push(r.u64()?);
            }
        }
        let codes = r.bytes(r.remaining())?.to_vec();
        let batch = Self::new(dict_id, codes, offsets)?;
        match group_size_log2 {
            Some(g) => batch.with_decoded_index(g, group_starts),
            None => Ok(batch),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Compressor, Decompressor, Dictionary, Scheme};
    use alloc::vec;

    #[test]
    fn random_access() {
        let dict = Dictionary::from_symbols(Scheme::U8, [&b"http://"[..], b"www.", b".com", b"/"]).unwrap();
        let c = Compressor::new(&dict).unwrap();
        let d = Decompressor::new(&dict).unwrap();
        let urls = ["http://www.example.com/", "", "http://a.b/c", "\u{1F980} not a url"];
        let batch = c.compress_strings(urls);
        assert_eq!(batch.len(), urls.len());
        for (i, url) in urls.iter().enumerate().rev() {
            let mut out = vec![];
            d.decompress_string(&batch, i, &mut out).unwrap();
            assert_eq!(out, url.as_bytes());
        }
        let (all, offsets) = d.decompress_strings(&batch).unwrap();
        assert_eq!(all, urls.concat().as_bytes());
        assert_eq!(offsets.len(), urls.len() + 1);

        let back = StringBatch::from_bytes(&batch.to_bytes()).unwrap();
        assert_eq!(back, batch);
        assert!(d.decompress_string(&batch, 99, &mut vec![]).is_err());
    }

    #[test]
    fn decoded_index() {
        let dict = Dictionary::from_symbols(Scheme::U8, [&b"ab"[..], b"c"]).unwrap();
        let c = Compressor::new(&dict).unwrap();
        let d = Decompressor::new(&dict).unwrap();
        let strings: Vec<alloc::string::String> = (0..100).map(|i| "abc".repeat(i % 7)).collect();
        let batch = c.compress_strings_with(&strings, StringOptions { decoded_index: true });
        let (g, starts) = batch.decoded_index().unwrap();
        assert_eq!(g, STRING_GROUP_SIZE_LOG2);
        assert_eq!(starts.len(), 100usize.div_ceil(32) + 1);
        let (all, offsets) = d.decompress_strings(&batch).unwrap();
        for (k, &s) in starts.iter().enumerate() {
            assert_eq!(u64::from(offsets[(k * 32).min(100)]), s);
        }
        assert_eq!(all.len() as u64, *starts.last().unwrap());
        let bytes = batch.to_bytes();
        assert_eq!(bytes[4], BATCH_VERSION_INDEXED);
        assert_eq!(StringBatch::from_bytes(&bytes).unwrap(), batch);
        // Without the option, nothing changes.
        let plain = c.compress_strings(&strings);
        assert!(plain.decoded_index().is_none());
        assert_eq!(plain.to_bytes()[4], BATCH_VERSION);
        // A wrong index is rejected when parsing or when decoding.
        let mut bad = starts.to_vec();
        bad[1] += 1;
        let wrong = plain.clone().with_decoded_index(STRING_GROUP_SIZE_LOG2, bad).unwrap();
        assert!(d.decompress_strings(&wrong).is_err());
        assert!(
            plain
                .clone()
                .with_decoded_index(STRING_GROUP_SIZE_LOG2, vec![0])
                .is_err()
        );
        assert!(plain.with_decoded_index(0, starts.to_vec()).is_err());
    }

    #[test]
    fn rejects_bad_offsets() {
        assert!(StringBatch::new(DictId(0), vec![1, 2], vec![0, 3]).is_err());
        assert!(StringBatch::new(DictId(0), vec![1, 2], vec![0, 2, 1]).is_err());
        assert!(StringBatch::new(DictId(0), vec![1, 2], vec![1, 2]).is_err());
        assert!(StringBatch::from_bytes(b"PRSC\x01").is_err());
    }
}
