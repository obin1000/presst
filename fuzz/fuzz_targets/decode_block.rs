#![no_main]
//! Arbitrary code streams: decoding must never panic and must agree with the
//! reference decoder.

use libfuzzer_sys::fuzz_target;
use presst::{reference, Decompressor};

#[path = "common.rs"]
mod common;

fuzz_target!(|input: (u16, &[u8])| {
    let (limit, codes) = input;
    let dict = common::fixed_dictionary();
    let dec = Decompressor::new(&dict).unwrap();
    let mut out = Vec::new();
    let _ = dec.decompress_block_into(codes, limit as usize, &mut out);
    let mut fast = Vec::new();
    match reference::decode_u8(&dict, codes) {
        Ok(expected) => {
            dec.decompress_codes_into(codes, &mut fast).unwrap();
            assert_eq!(fast, expected);
        }
        Err(_) => assert!(dec.decompress_codes_into(codes, &mut fast).is_err()),
    }
});
