//! Shared helpers for the fuzz targets. Each target uses a subset.
#![allow(dead_code)]

use presst::{Dictionary, DictionaryBuilder, Scheme};

/// A fixed dictionary with short and long symbols.
pub fn fixed_dictionary() -> Dictionary {
    Dictionary::from_symbols(
        Scheme::U8,
        [&b"the "[..], b"and", b"ing ", b"http://", b"e", b" ", b"th", b"in", b"abcdefgh", b"\x00\x00"],
    )
    .unwrap()
}

/// Builds a valid dictionary from arbitrary symbols, skipping invalid ones.
pub fn dictionary_from(symbols: &[Vec<u8>]) -> Dictionary {
    let mut b = DictionaryBuilder::new(Scheme::U8).unwrap();
    for s in symbols {
        if b.is_full() {
            break;
        }
        b.push_lossy(s);
    }
    b.build()
}
