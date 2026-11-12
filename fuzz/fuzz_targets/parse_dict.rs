#![no_main]
//! Arbitrary bytes as a `.pdict` file: parsing must never panic, and a parsed
//! dictionary must serialize back to the same bytes.

use libfuzzer_sys::fuzz_target;
use presst::{Compressor, Decompressor, Dictionary};

fuzz_target!(|data: &[u8]| {
    if let Ok(dict) = Dictionary::from_bytes(data) {
        assert_eq!(Dictionary::from_bytes(&dict.to_bytes()).unwrap(), dict);
        Compressor::new(&dict).unwrap();
        Decompressor::new(&dict).unwrap();
    }
});
