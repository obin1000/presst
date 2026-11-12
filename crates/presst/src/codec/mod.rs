//! Prepared compressors and decompressors.

pub(crate) mod u8;

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;

use self::u8::{U8Decoder, U8Encoder};
use crate::frame::{self, FrameOptions};
use crate::strings::{StringBatch, StringOptions};
use crate::{DictId, Dictionary, Error, Result, Scheme};

/// Encoder tables per scheme. Dispatch happens once per call, never per byte.
enum EncTables {
    U8(U8Encoder),
}

/// Decoder tables per scheme.
enum DecTables {
    U8(U8Decoder),
}

struct EncInner {
    id: DictId,
    scheme: Scheme,
    tables: EncTables,
}

struct DecInner {
    id: DictId,
    scheme: Scheme,
    tables: DecTables,
}

/// Compresses data with one pretrained dictionary.
///
/// Creating a `Compressor` builds the encoder tables once (about 150 KB for
/// U8); after that, compressing needs no training and no per-block setup.
/// Cloning is cheap (the tables are shared), and the type is `Send + Sync`.
#[derive(Clone)]
pub struct Compressor {
    inner: Arc<EncInner>,
}

impl fmt::Debug for Compressor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Compressor")
            .field("dict", &self.inner.id)
            .field("scheme", &self.inner.scheme)
            .finish()
    }
}

impl Compressor {
    /// Prepares the encoder tables for `dict`.
    pub fn new(dict: &Dictionary) -> Result<Self> {
        let tables = match dict.scheme() {
            Scheme::U8 => EncTables::U8(U8Encoder::new(dict)?),
            other => return Err(Error::UnsupportedScheme(other.as_u8())),
        };
        Ok(Self {
            inner: Arc::new(EncInner {
                id: dict.id(),
                scheme: dict.scheme(),
                tables,
            }),
        })
    }

    /// Id of the dictionary this compressor uses.
    #[must_use]
    pub fn dict_id(&self) -> DictId {
        self.inner.id
    }

    /// Code scheme of the dictionary.
    #[must_use]
    pub fn scheme(&self) -> Scheme {
        self.inner.scheme
    }

    /// Compresses `input` into a self-describing frame with default options
    /// (1 MiB blocks, content size and checksum included).
    #[must_use]
    pub fn compress(&self, input: &[u8]) -> Vec<u8> {
        self.compress_with(input, FrameOptions::default())
    }

    /// Compresses `input` into a frame with the given options.
    #[must_use]
    pub fn compress_with(&self, input: &[u8], options: FrameOptions) -> Vec<u8> {
        let mut out = Vec::new();
        frame::compress_frame(self, input, options, &mut out);
        out
    }

    /// Appends the raw code stream for `input` to `out`, without any framing.
    /// Returns the number of bytes appended.
    ///
    /// The caller must store the decoded length (or an upper bound) and the
    /// [`DictId`] to decode it again with [`Decompressor::decompress_block_into`].
    pub fn compress_block_into(&self, input: &[u8], out: &mut Vec<u8>) -> usize {
        let before = out.len();
        match &self.inner.tables {
            EncTables::U8(enc) => enc.encode(input, out),
        }
        out.len() - before
    }

    /// Compresses many short strings into one [`StringBatch`] with per-string
    /// random access.
    ///
    /// # Panics
    /// Panics if the total compressed size exceeds `u32::MAX` bytes.
    pub fn compress_strings<I, S>(&self, strings: I) -> StringBatch
    where
        I: IntoIterator<Item = S>,
        S: AsRef<[u8]>,
    {
        self.compress_strings_with(strings, StringOptions::default())
    }

    /// Like [`compress_strings`](Self::compress_strings), with options (for
    /// example a decoded index for GPU decoding).
    ///
    /// # Panics
    /// Panics if the total compressed size exceeds `u32::MAX` bytes.
    pub fn compress_strings_with<I, S>(&self, strings: I, options: StringOptions) -> StringBatch
    where
        I: IntoIterator<Item = S>,
        S: AsRef<[u8]>,
    {
        let iter = strings.into_iter();
        let mut codes = Vec::new();
        let mut offsets = Vec::with_capacity(iter.size_hint().0 + 1);
        let mut decoded_lens = Vec::new();
        offsets.push(0);
        for s in iter {
            let s = s.as_ref();
            self.compress_block_into(s, &mut codes);
            offsets.push(u32::try_from(codes.len()).expect("string batch larger than 4 GiB"));
            if options.decoded_index {
                decoded_lens.push(s.len());
            }
        }
        let mut batch = StringBatch::from_parts(self.inner.id, codes, offsets);
        if options.decoded_index {
            batch.set_decoded_index(&decoded_lens);
        }
        batch
    }
}

/// Decompresses data produced with one pretrained dictionary.
///
/// Cloning is cheap (the tables are shared), and the type is `Send + Sync`.
/// All decoding methods validate their input and return an [`Error`] instead
/// of panicking on corrupt or malicious data.
#[derive(Clone)]
pub struct Decompressor {
    inner: Arc<DecInner>,
}

impl fmt::Debug for Decompressor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Decompressor")
            .field("dict", &self.inner.id)
            .field("scheme", &self.inner.scheme)
            .finish()
    }
}

impl Decompressor {
    /// Prepares the decoder tables for `dict`.
    pub fn new(dict: &Dictionary) -> Result<Self> {
        let tables = match dict.scheme() {
            Scheme::U8 => DecTables::U8(U8Decoder::new(dict)),
            other => return Err(Error::UnsupportedScheme(other.as_u8())),
        };
        Ok(Self {
            inner: Arc::new(DecInner {
                id: dict.id(),
                scheme: dict.scheme(),
                tables,
            }),
        })
    }

    /// Id of the dictionary this decompressor uses.
    #[must_use]
    pub fn dict_id(&self) -> DictId {
        self.inner.id
    }

    /// Code scheme of the dictionary.
    #[must_use]
    pub fn scheme(&self) -> Scheme {
        self.inner.scheme
    }

    /// Decompresses a complete frame.
    pub fn decompress(&self, frame: &[u8]) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        self.decompress_into(frame, &mut out)?;
        Ok(out)
    }

    /// Decompresses a complete frame, appending to `out`. Returns the number of
    /// bytes appended. On error, `out` keeps its original length.
    pub fn decompress_into(&self, frame: &[u8], out: &mut Vec<u8>) -> Result<usize> {
        let before = out.len();
        let res = frame::decompress_frame(self, frame, out);
        if res.is_err() {
            out.truncate(before);
        }
        res
    }

    /// Decodes a raw code stream that decodes to exactly `decoded_len` bytes,
    /// appending to `out`.
    pub fn decompress_block_into(&self, codes: &[u8], decoded_len: usize, out: &mut Vec<u8>) -> Result<()> {
        let n = self.decode_limited(codes, out, decoded_len)?;
        if n != decoded_len {
            out.truncate(out.len() - n);
            return Err(Error::Corrupt("decoded size does not match"));
        }
        Ok(())
    }

    /// Decodes a raw code stream of unknown decoded length, appending to `out`.
    /// Returns the number of bytes appended.
    pub fn decompress_codes_into(&self, codes: &[u8], out: &mut Vec<u8>) -> Result<usize> {
        let limit = codes.len().saturating_mul(self.max_expansion());
        self.decode_limited(codes, out, limit)
    }

    /// Decodes string `index` of `batch`, appending it to `out`.
    pub fn decompress_string(&self, batch: &StringBatch, index: usize, out: &mut Vec<u8>) -> Result<()> {
        self.check_dict(batch.dict_id())?;
        let codes = batch.codes(index).ok_or(Error::Corrupt("string index out of range"))?;
        self.decompress_codes_into(codes, out).map(|_| ())
    }

    /// Decodes every string of `batch` into one buffer. Returns the bytes and
    /// `len + 1` offsets into them.
    pub fn decompress_strings(&self, batch: &StringBatch) -> Result<(Vec<u8>, Vec<u32>)> {
        self.check_dict(batch.dict_id())?;
        let mut data = Vec::with_capacity(batch.codes_len() * 3);
        let mut offsets = Vec::with_capacity(batch.len() + 1);
        offsets.push(0);
        for i in 0..batch.len() {
            let codes = batch.codes(i).ok_or(Error::Corrupt("string index out of range"))?;
            self.decompress_codes_into(codes, &mut data)?;
            offsets.push(u32::try_from(data.len()).map_err(|_| Error::Corrupt("decoded batch larger than 4 GiB"))?);
        }
        // A stored decoded index must match what the strings decode to.
        if let Some((g, starts)) = batch.decoded_index() {
            let g = 1usize << g;
            let ok = starts
                .iter()
                .enumerate()
                .all(|(k, &s)| u64::from(offsets[(k * g).min(batch.len())]) == s);
            if !ok {
                return Err(Error::Corrupt("decoded index does not match the strings"));
            }
        }
        Ok((data, offsets))
    }

    pub(crate) fn check_dict(&self, id: DictId) -> Result<()> {
        if id == self.inner.id {
            Ok(())
        } else {
            Err(Error::DictionaryMismatch {
                expected: id,
                actual: self.inner.id,
            })
        }
    }

    /// Maximum bytes one code byte can decode to.
    pub(crate) fn max_expansion(&self) -> usize {
        match self.inner.scheme {
            Scheme::U8 => 8,
            Scheme::U16 | Scheme::Hybrid8x16 => 16,
        }
    }

    pub(crate) fn decode_limited(&self, codes: &[u8], out: &mut Vec<u8>, limit: usize) -> Result<usize> {
        match &self.inner.tables {
            DecTables::U8(dec) => dec.decode(codes, out, limit),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn types_are_send_sync() {
        assert_send_sync::<Compressor>();
        assert_send_sync::<Decompressor>();
    }
}
