#![no_main]
//! Arbitrary dictionary and input: the fast encoder must match the reference,
//! and every frame must round-trip.

use libfuzzer_sys::fuzz_target;
use presst::{reference, Compressor, Decompressor, FrameOptions};

#[path = "common.rs"]
mod common;

fuzz_target!(|input: (Vec<Vec<u8>>, Vec<u8>, bool)| {
    let (symbols, data, small_blocks) = input;
    let dict = common::dictionary_from(&symbols);
    let comp = Compressor::new(&dict).unwrap();
    let dec = Decompressor::new(&dict).unwrap();

    let mut codes = Vec::new();
    comp.compress_block_into(&data, &mut codes);
    assert_eq!(codes, reference::encode_u8(&dict, &data));

    let opts = FrameOptions { block_size_log2: if small_blocks { 10 } else { 20 }, ..FrameOptions::default() };
    let frame = comp.compress_with(&data, opts);
    assert_eq!(dec.decompress(&frame).unwrap(), data);
});
