//! Looking dictionaries up by id when decoding.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::frame::FrameHeader;
use crate::{Compressor, Decompressor, DictId, Dictionary, Error, Result, StringBatch};

/// A set of dictionaries, indexed by [`DictId`].
///
/// Frames only store the id of their dictionary; a registry resolves it and
/// keeps the prepared decoder tables hot across calls.
#[derive(Debug, Clone, Default)]
pub struct Registry {
    entries: BTreeMap<DictId, (Dictionary, Decompressor)>,
}

impl Registry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a dictionary and returns its id. Adding the same dictionary twice
    /// is a no-op.
    pub fn insert(&mut self, dict: Dictionary) -> Result<DictId> {
        let id = dict.id();
        if let alloc::collections::btree_map::Entry::Vacant(slot) = self.entries.entry(id) {
            let dec = Decompressor::new(&dict)?;
            slot.insert((dict, dec));
        }
        Ok(id)
    }

    /// The dictionary with this id.
    #[must_use]
    pub fn get(&self, id: DictId) -> Option<&Dictionary> {
        self.entries.get(&id).map(|(d, _)| d)
    }

    /// A decompressor for the dictionary with this id.
    pub fn decompressor(&self, id: DictId) -> Result<&Decompressor> {
        self.entries
            .get(&id)
            .map(|(_, d)| d)
            .ok_or(Error::UnknownDictionary(id))
    }

    /// A new compressor for the dictionary with this id.
    pub fn compressor(&self, id: DictId) -> Result<Compressor> {
        let (dict, _) = self.entries.get(&id).ok_or(Error::UnknownDictionary(id))?;
        Compressor::new(dict)
    }

    /// Ids of all registered dictionaries.
    pub fn ids(&self) -> impl Iterator<Item = DictId> + '_ {
        self.entries.keys().copied()
    }

    /// Number of registered dictionaries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Decompresses a frame with whichever registered dictionary it needs.
    pub fn decompress(&self, frame: &[u8]) -> Result<Vec<u8>> {
        let (header, _) = FrameHeader::parse(frame)?;
        self.decompressor(header.dict_id)?.decompress(frame)
    }

    /// Decodes one string of a batch with the dictionary it needs.
    pub fn decompress_string(&self, batch: &StringBatch, index: usize, out: &mut Vec<u8>) -> Result<()> {
        self.decompressor(batch.dict_id())?.decompress_string(batch, index, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Scheme;

    #[test]
    fn resolves_by_id() {
        let a = Dictionary::from_symbols(Scheme::U8, [&b"alpha"[..]]).unwrap();
        let b = Dictionary::from_symbols(Scheme::U8, [&b"beta"[..]]).unwrap();
        let mut reg = Registry::new();
        let ida = reg.insert(a.clone()).unwrap();
        reg.insert(b.clone()).unwrap();
        reg.insert(a.clone()).unwrap();
        assert_eq!(reg.len(), 2);
        let fa = Compressor::new(&a).unwrap().compress(b"alpha alpha");
        let fb = reg.compressor(b.id()).unwrap().compress(b"beta beta");
        assert_eq!(reg.decompress(&fa).unwrap(), b"alpha alpha");
        assert_eq!(reg.decompress(&fb).unwrap(), b"beta beta");
        assert_eq!(reg.get(ida), Some(&a));
        let empty = Registry::new();
        assert_eq!(empty.decompress(&fa), Err(Error::UnknownDictionary(ida)));
    }
}
