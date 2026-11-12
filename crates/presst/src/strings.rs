//! Batches of individually compressed strings, for columnar use.

use alloc::vec::Vec;

use crate::util::{Reader, put_u32, put_u64};
use crate::{DictId, Error, Result};

const BATCH_MAGIC: [u8; 4] = *b"PRSC";
const BATCH_VERSION: u8 = 1;

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
}

impl StringBatch {
    pub(crate) fn from_parts(dict_id: DictId, codes: Vec<u8>, offsets: Vec<u32>) -> Self {
        debug_assert_eq!(offsets.first(), Some(&0));
        Self {
            dict_id,
            codes,
            offsets,
        }
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
        })
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
        let mut out = Vec::with_capacity(17 + self.offsets.len() * 4 + self.codes.len());
        out.extend_from_slice(&BATCH_MAGIC);
        out.push(BATCH_VERSION);
        put_u64(&mut out, self.dict_id.0);
        put_u32(&mut out, self.len() as u32);
        for &o in &self.offsets {
            put_u32(&mut out, o);
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
        if version != BATCH_VERSION {
            return Err(Error::UnsupportedVersion(version as u16));
        }
        let dict_id = DictId(r.u64()?);
        let count = r.u32()? as usize;
        if count.saturating_add(1).saturating_mul(4) > r.remaining() {
            return Err(Error::Truncated);
        }
        let mut offsets = Vec::with_capacity(count + 1);
        for _ in 0..=count {
            offsets.push(r.u32()?);
        }
        let codes = r.bytes(r.remaining())?.to_vec();
        Self::new(dict_id, codes, offsets)
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
    fn rejects_bad_offsets() {
        assert!(StringBatch::new(DictId(0), vec![1, 2], vec![0, 3]).is_err());
        assert!(StringBatch::new(DictId(0), vec![1, 2], vec![0, 2, 1]).is_err());
        assert!(StringBatch::new(DictId(0), vec![1, 2], vec![1, 2]).is_err());
        assert!(StringBatch::from_bytes(b"PRSC\x01").is_err());
    }
}
