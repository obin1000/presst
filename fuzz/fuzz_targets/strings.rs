#![no_main]
//! String batches round-trip, and arbitrary `PRSC` bytes never panic.

use libfuzzer_sys::fuzz_target;
use presst::{Compressor, Decompressor, StringBatch};

#[path = "common.rs"]
mod common;

fuzz_target!(|input: (Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<u8>)| {
    let (symbols, strings, raw) = input;
    let dict = common::dictionary_from(&symbols);
    let comp = Compressor::new(&dict).unwrap();
    let dec = Decompressor::new(&dict).unwrap();
    let batch = StringBatch::from_bytes(&comp.compress_strings(&strings).to_bytes()).unwrap();
    for (i, s) in strings.iter().enumerate() {
        let mut out = Vec::new();
        dec.decompress_string(&batch, i, &mut out).unwrap();
        assert_eq!(&out, s);
    }
    if let Ok(batch) = StringBatch::from_bytes(&raw) {
        let _ = dec.decompress_strings(&batch);
    }
});
