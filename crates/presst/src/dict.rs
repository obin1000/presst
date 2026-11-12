//! Dictionaries: the pretrained symbol tables and their `.pdict` file format.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use xxhash_rust::xxh3::xxh3_64;

use crate::scheme::Scheme;
use crate::util::{Reader, put_u16, put_u32, put_u64};
use crate::{Error, Result};

/// Magic bytes at the start of a `.pdict` file.
pub(crate) const DICT_MAGIC: [u8; 4] = *b"PRSD";
/// Current `.pdict` format version.
pub(crate) const DICT_VERSION: u16 = 1;

/// Content-addressed identifier of a dictionary.
///
/// It is the xxh3-64 hash of the scheme, maximum symbol length, symbol count,
/// symbol lengths and symbol bytes. Metadata is not included, so renaming or
/// re-saving a dictionary keeps its id.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DictId(pub u64);

impl fmt::Display for DictId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

impl fmt::Debug for DictId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DictId({:016x})", self.0)
    }
}

/// Descriptive metadata stored in a `.pdict` file. It does not affect the
/// [`DictId`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Metadata {
    /// Human-readable name, for example `"tpch-comment"`.
    pub name: Option<String>,
    /// Trainer name and version that produced the dictionary.
    pub trainer: Option<String>,
    /// Random seed used for training.
    pub seed: Option<u64>,
    /// Hash of the training corpus manifest.
    pub corpus_hash: Option<u64>,
}

const TAG_NAME: u8 = 1;
const TAG_TRAINER: u8 = 2;
const TAG_SEED: u8 = 3;
const TAG_CORPUS_HASH: u8 = 4;

impl Metadata {
    fn write(&self, out: &mut Vec<u8>) {
        let mut tlv = |tag: u8, data: &[u8]| {
            out.push(tag);
            put_u16(out, u16::try_from(data.len()).unwrap_or(u16::MAX));
            out.extend_from_slice(&data[..data.len().min(u16::MAX as usize)]);
        };
        if let Some(n) = &self.name {
            tlv(TAG_NAME, n.as_bytes());
        }
        if let Some(t) = &self.trainer {
            tlv(TAG_TRAINER, t.as_bytes());
        }
        if let Some(s) = self.seed {
            tlv(TAG_SEED, &s.to_le_bytes());
        }
        if let Some(h) = self.corpus_hash {
            tlv(TAG_CORPUS_HASH, &h.to_le_bytes());
        }
    }

    fn read(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader::new(bytes);
        let mut meta = Metadata::default();
        while r.remaining() > 0 {
            let tag = r.u8()?;
            let len = r.u16()? as usize;
            let data = r.bytes(len)?;
            let text = || String::from_utf8(data.to_vec()).map_err(|_| Error::Corrupt("metadata is not UTF-8"));
            let num = || -> Result<u64> {
                let a: [u8; 8] = data.try_into().map_err(|_| Error::Corrupt("bad metadata integer"))?;
                Ok(u64::from_le_bytes(a))
            };
            match tag {
                TAG_NAME => meta.name = Some(text()?),
                TAG_TRAINER => meta.trainer = Some(text()?),
                TAG_SEED => meta.seed = Some(num()?),
                TAG_CORPUS_HASH => meta.corpus_hash = Some(num()?),
                _ => {} // unknown tags are skipped for forward compatibility
            }
        }
        Ok(meta)
    }
}

/// A pretrained symbol table.
///
/// Symbols are numbered by their position: symbol `i` is emitted as code `i`.
/// Build one with [`Dictionary::from_symbols`] or [`DictionaryBuilder`], load
/// one with [`Dictionary::from_bytes`], or train one with `presst-train`.
#[derive(Clone, PartialEq, Eq)]
pub struct Dictionary {
    scheme: Scheme,
    max_sym_len: u8,
    lens: Vec<u8>,
    offsets: Vec<u32>,
    bytes: Vec<u8>,
    id: DictId,
    meta: Metadata,
}

impl fmt::Debug for Dictionary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Dictionary")
            .field("id", &self.id)
            .field("scheme", &self.scheme)
            .field("symbols", &self.len())
            .field("meta", &self.meta)
            .finish_non_exhaustive()
    }
}

impl Dictionary {
    /// Builds a dictionary from symbols in code order.
    ///
    /// Returns [`Error::InvalidDictionary`] if any symbol violates the
    /// scheme's constraints (see [`DictionaryBuilder::try_push`]). Use
    /// [`DictionaryBuilder::push_lossy`] to skip conflicting symbols instead.
    pub fn from_symbols<I, S>(scheme: Scheme, symbols: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<[u8]>,
    {
        let mut b = DictionaryBuilder::new(scheme)?;
        for s in symbols {
            b.try_push(s.as_ref())
                .map_err(|e| Error::InvalidDictionary(e.as_str()))?;
        }
        Ok(b.build())
    }

    /// The code scheme of this dictionary.
    #[must_use]
    pub fn scheme(&self) -> Scheme {
        self.scheme
    }

    /// Maximum symbol length allowed by the scheme.
    #[must_use]
    pub fn max_symbol_len(&self) -> usize {
        self.max_sym_len as usize
    }

    /// The content-addressed id of this dictionary.
    #[must_use]
    pub fn id(&self) -> DictId {
        self.id
    }

    /// Number of symbols.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lens.len()
    }

    /// Whether the dictionary has no symbols. Such a dictionary is valid: every
    /// byte is escaped.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lens.is_empty()
    }

    /// The symbol emitted as code `code`.
    ///
    /// # Panics
    /// Panics if `code >= self.len()`.
    #[must_use]
    pub fn symbol(&self, code: usize) -> &[u8] {
        let start = self.offsets[code] as usize;
        &self.bytes[start..start + self.lens[code] as usize]
    }

    /// Iterates over the symbols in code order.
    pub fn symbols(&self) -> impl ExactSizeIterator<Item = &[u8]> + '_ {
        (0..self.len()).map(move |i| self.symbol(i))
    }

    /// Symbol lengths in code order.
    #[must_use]
    pub fn symbol_lens(&self) -> &[u8] {
        &self.lens
    }

    /// Descriptive metadata.
    #[must_use]
    pub fn metadata(&self) -> &Metadata {
        &self.meta
    }

    /// Replaces the metadata. The [`DictId`] does not change.
    #[must_use]
    pub fn with_metadata(mut self, meta: Metadata) -> Self {
        self.meta = meta;
        self
    }

    /// Serializes the dictionary to the `.pdict` format.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(32 + self.lens.len() + self.bytes.len());
        out.extend_from_slice(&DICT_MAGIC);
        put_u16(&mut out, DICT_VERSION);
        out.push(self.scheme.as_u8());
        out.push(self.max_sym_len);
        put_u16(&mut out, 0); // flags
        put_u32(&mut out, self.lens.len() as u32);
        let mut meta = Vec::new();
        self.meta.write(&mut meta);
        put_u32(&mut out, meta.len() as u32);
        out.extend_from_slice(&meta);
        out.extend_from_slice(&self.lens);
        out.extend_from_slice(&self.bytes);
        let checksum = xxh3_64(&out);
        put_u64(&mut out, checksum);
        out
    }

    /// Parses and validates a `.pdict` file.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 8 {
            return Err(Error::Truncated);
        }
        let (body, checksum) = data.split_at(data.len() - 8);
        let mut r = Reader::new(body);
        if r.array::<4>()? != DICT_MAGIC {
            return Err(Error::BadMagic);
        }
        let version = r.u16()?;
        if version != DICT_VERSION {
            return Err(Error::UnsupportedVersion(version));
        }
        if u64::from_le_bytes(checksum.try_into().map_err(|_| Error::Truncated)?) != xxh3_64(body) {
            return Err(Error::ChecksumMismatch);
        }
        let scheme = Scheme::from_u8(r.u8()?)?;
        let max_sym_len = r.u8()?;
        if max_sym_len as usize != scheme.max_symbol_len() {
            return Err(Error::InvalidDictionary("max symbol length does not match scheme"));
        }
        let flags = r.u16()?;
        if flags != 0 {
            return Err(Error::Corrupt("unknown dictionary flags"));
        }
        let count = r.u32()? as usize;
        if count > scheme.max_symbols() {
            return Err(Error::InvalidDictionary("too many symbols"));
        }
        let meta_len = r.u32()? as usize;
        let meta = Metadata::read(r.bytes(meta_len)?)?;
        let lens = r.bytes(count)?;
        let total: usize = lens.iter().map(|&l| l as usize).sum();
        let bytes = r.bytes(total)?;
        if r.remaining() != 0 {
            return Err(Error::Corrupt("trailing bytes in dictionary"));
        }
        let mut b = DictionaryBuilder::new(scheme)?;
        let mut pos = 0;
        for &l in lens {
            let sym = &bytes[pos..pos + l as usize];
            pos += l as usize;
            b.try_push(sym).map_err(|e| Error::InvalidDictionary(e.as_str()))?;
        }
        Ok(b.build().with_metadata(meta))
    }

    pub(crate) fn compute_id(scheme: Scheme, max_sym_len: u8, lens: &[u8], bytes: &[u8]) -> DictId {
        let mut buf = Vec::with_capacity(6 + lens.len() + bytes.len());
        buf.push(scheme.as_u8());
        buf.push(max_sym_len);
        put_u32(&mut buf, lens.len() as u32);
        buf.extend_from_slice(lens);
        buf.extend_from_slice(bytes);
        DictId(xxh3_64(&buf))
    }
}

/// Why a symbol could not be added to a [`DictionaryBuilder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PushError {
    /// The dictionary already holds the maximum number of symbols.
    Full,
    /// The symbol is empty or longer than the scheme allows.
    BadLength,
    /// The symbol is already in the dictionary.
    Duplicate,
    /// U8 only: another symbol of 3 or more bytes starts with the same 3 bytes.
    /// The encoder looks long symbols up by their 3-byte prefix, so each
    /// prefix can map to at most one of them.
    PrefixConflict {
        /// Code of the conflicting symbol.
        code: usize,
    },
}

impl PushError {
    fn as_str(self) -> &'static str {
        match self {
            PushError::Full => "too many symbols",
            PushError::BadLength => "symbol length out of range",
            PushError::Duplicate => "duplicate symbol",
            PushError::PrefixConflict { .. } => "two long symbols share a 3-byte prefix",
        }
    }
}

impl fmt::Display for PushError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl core::error::Error for PushError {}

/// Incrementally builds a valid [`Dictionary`], enforcing the scheme's
/// constraints on every push.
#[derive(Debug, Clone)]
pub struct DictionaryBuilder {
    scheme: Scheme,
    lens: Vec<u8>,
    bytes: Vec<u8>,
    seen: BTreeSet<Vec<u8>>,
    prefixes: BTreeMap<[u8; 3], usize>,
}

impl DictionaryBuilder {
    /// Creates an empty builder for `scheme`.
    pub fn new(scheme: Scheme) -> Result<Self> {
        // Only implemented schemes can be built.
        Scheme::from_u8(scheme.as_u8())?;
        Ok(Self {
            scheme,
            lens: Vec::new(),
            bytes: Vec::new(),
            seen: BTreeSet::new(),
            prefixes: BTreeMap::new(),
        })
    }

    /// Number of symbols added so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lens.len()
    }

    /// Whether no symbols were added yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lens.is_empty()
    }

    /// Whether the builder holds the maximum number of symbols.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.lens.len() >= self.scheme.max_symbols()
    }

    /// Checks whether `sym` could be added, without adding it.
    pub fn check(&self, sym: &[u8]) -> Result<(), PushError> {
        if self.is_full() {
            return Err(PushError::Full);
        }
        if sym.is_empty() || sym.len() > self.scheme.max_symbol_len() {
            return Err(PushError::BadLength);
        }
        if self.seen.contains(sym) {
            return Err(PushError::Duplicate);
        }
        if sym.len() >= 3 {
            if let Some(&code) = self.prefixes.get(&prefix3(sym)) {
                return Err(PushError::PrefixConflict { code });
            }
        }
        Ok(())
    }

    /// Adds `sym` as the next code, or explains why it cannot be added.
    pub fn try_push(&mut self, sym: &[u8]) -> Result<usize, PushError> {
        self.check(sym)?;
        let code = self.lens.len();
        if sym.len() >= 3 {
            self.prefixes.insert(prefix3(sym), code);
        }
        self.seen.insert(sym.to_vec());
        self.lens.push(sym.len() as u8);
        self.bytes.extend_from_slice(sym);
        Ok(code)
    }

    /// Adds `sym` if it satisfies the constraints and returns whether it was added.
    pub fn push_lossy(&mut self, sym: &[u8]) -> bool {
        self.try_push(sym).is_ok()
    }

    /// Finishes the dictionary.
    #[must_use]
    pub fn build(self) -> Dictionary {
        let max_sym_len = self.scheme.max_symbol_len() as u8;
        let mut offsets = Vec::with_capacity(self.lens.len());
        let mut pos = 0u32;
        for &l in &self.lens {
            offsets.push(pos);
            pos += l as u32;
        }
        let id = Dictionary::compute_id(self.scheme, max_sym_len, &self.lens, &self.bytes);
        Dictionary {
            scheme: self.scheme,
            max_sym_len,
            lens: self.lens,
            offsets,
            bytes: self.bytes,
            id,
            meta: Metadata::default(),
        }
    }
}

fn prefix3(sym: &[u8]) -> [u8; 3] {
    [sym[0], sym[1], sym[2]]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Dictionary {
        Dictionary::from_symbols(Scheme::U8, [&b"abc"[..], b"ab", b"x", b"hello wo"]).unwrap()
    }

    #[test]
    fn roundtrip_bytes() {
        let d = sample().with_metadata(Metadata {
            name: Some("test".into()),
            trainer: Some("unit".into()),
            seed: Some(42),
            corpus_hash: None,
        });
        let bytes = d.to_bytes();
        let back = Dictionary::from_bytes(&bytes).unwrap();
        assert_eq!(back, d);
        assert_eq!(back.id(), d.id());
    }

    #[test]
    fn id_ignores_metadata() {
        let a = sample();
        let b = sample().with_metadata(Metadata {
            name: Some("other".into()),
            ..Metadata::default()
        });
        assert_eq!(a.id(), b.id());
    }

    #[test]
    fn id_depends_on_order() {
        let a = Dictionary::from_symbols(Scheme::U8, [&b"a"[..], b"b"]).unwrap();
        let b = Dictionary::from_symbols(Scheme::U8, [&b"b"[..], b"a"]).unwrap();
        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn constraints() {
        let mut b = DictionaryBuilder::new(Scheme::U8).unwrap();
        assert_eq!(b.try_push(b""), Err(PushError::BadLength));
        assert_eq!(b.try_push(b"123456789"), Err(PushError::BadLength));
        assert_eq!(b.try_push(b"abcd"), Ok(0));
        assert_eq!(b.try_push(b"abcd"), Err(PushError::Duplicate));
        assert_eq!(b.try_push(b"abce"), Err(PushError::PrefixConflict { code: 0 }));
        assert_eq!(b.try_push(b"abc"), Err(PushError::PrefixConflict { code: 0 }));
        assert_eq!(b.try_push(b"ab"), Ok(1));
        for i in 0..253u8 {
            assert!(b.push_lossy(&[i, 0xFE]) || i == b'a');
        }
        assert!(b.is_full());
        assert_eq!(b.try_push(b"zz"), Err(PushError::Full));
    }

    #[test]
    fn rejects_corruption() {
        let mut bytes = sample().to_bytes();
        let n = bytes.len();
        bytes[10] ^= 1;
        assert_eq!(Dictionary::from_bytes(&bytes), Err(Error::ChecksumMismatch));
        assert_eq!(Dictionary::from_bytes(&bytes[..n - 3]), Err(Error::ChecksumMismatch));
        assert_eq!(Dictionary::from_bytes(b"PRS"), Err(Error::Truncated));
        let mut bad_magic = sample().to_bytes();
        bad_magic[0] = b'X';
        assert_eq!(Dictionary::from_bytes(&bad_magic), Err(Error::BadMagic));
        assert!(Dictionary::from_bytes(&[0u8; 64]).is_err());
    }
}
