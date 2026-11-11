#![no_main]
//! Arbitrary bytes as a frame: decoding must never panic, and a frame that
//! decodes must decode identically through the streaming reader.

use std::io::Read;

use libfuzzer_sys::fuzz_target;
use presst::{io::FrameReader, Decompressor};

#[path = "common.rs"]
mod common;

fuzz_target!(|data: &[u8]| {
    let dec = Decompressor::new(&common::fixed_dictionary()).unwrap();
    let one_shot = dec.decompress(data);
    if let Ok(reader) = FrameReader::new(dec, data) {
        let mut streamed = Vec::new();
        let res = reader.take(1 << 24).read_to_end(&mut streamed);
        if let Ok(expected) = &one_shot {
            assert!(res.is_ok());
            assert_eq!(&streamed, expected);
        }
    }
});
