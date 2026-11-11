//! Property tests: the fast kernels must match the reference implementation,
//! round-trip any input, and never panic on arbitrary data.

use presst::{Compressor, Decompressor, Dictionary, DictionaryBuilder, FrameOptions, Scheme, StringBatch, reference};
use proptest::prelude::*;

/// A small alphabet makes symbols actually match the generated inputs.
fn alphabet_byte() -> impl Strategy<Value = u8> {
    prop_oneof![
        8 => prop::sample::select(b"abcde ".to_vec()),
        1 => any::<u8>(),
    ]
}

fn symbol() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(alphabet_byte(), 1..=8)
}

/// Random valid U8 dictionaries: conflicting symbols are skipped.
fn dictionary() -> impl Strategy<Value = Dictionary> {
    prop::collection::vec(symbol(), 0..300).prop_map(|syms| {
        let mut b = DictionaryBuilder::new(Scheme::U8).unwrap();
        for s in &syms {
            if b.is_full() {
                break;
            }
            b.push_lossy(s);
        }
        b.build()
    })
}

/// Inputs mixing dictionary symbols with random bytes, so both matches and
/// escapes occur, including at block and tail boundaries.
fn input_for(dict: &Dictionary) -> impl Strategy<Value = Vec<u8>> + use<> {
    let syms: Vec<Vec<u8>> = dict.symbols().map(<[u8]>::to_vec).collect();
    let pieces = if syms.is_empty() {
        prop::collection::vec(alphabet_byte(), 1..4).boxed()
    } else {
        prop_oneof![
            3 => prop::sample::select(syms),
            1 => prop::collection::vec(alphabet_byte(), 1..4),
        ]
        .boxed()
    };
    prop::collection::vec(pieces, 0..200).prop_map(|p| p.concat())
}

fn dict_and_input() -> impl Strategy<Value = (Dictionary, Vec<u8>)> {
    dictionary().prop_flat_map(|d| {
        let input = input_for(&d);
        (Just(d), input)
    })
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: if cfg!(miri) { 8 } else { 512 },
        ..ProptestConfig::default()
    })]

    #[test]
    fn fast_encoder_matches_reference((dict, input) in dict_and_input()) {
        let comp = Compressor::new(&dict).unwrap();
        let mut codes = Vec::new();
        comp.compress_block_into(&input, &mut codes);
        prop_assert_eq!(&codes, &reference::encode_u8(&dict, &input));
    }

    #[test]
    fn block_roundtrip((dict, input) in dict_and_input()) {
        let comp = Compressor::new(&dict).unwrap();
        let dec = Decompressor::new(&dict).unwrap();
        let mut codes = vec![0xAA; 3]; // appending must keep existing bytes
        comp.compress_block_into(&input, &mut codes);
        let mut out = vec![0xBB];
        dec.decompress_block_into(&codes[3..], input.len(), &mut out).unwrap();
        prop_assert_eq!(&out[1..], &input[..]);
        prop_assert_eq!(out[0], 0xBB);
        let mut out2 = Vec::new();
        prop_assert_eq!(dec.decompress_codes_into(&codes[3..], &mut out2).unwrap(), input.len());
        prop_assert_eq!(out2, input);
    }

    #[test]
    fn frame_roundtrip((dict, input) in dict_and_input(), checksum: bool, content_size: bool) {
        let comp = Compressor::new(&dict).unwrap();
        let dec = Decompressor::new(&dict).unwrap();
        let opts = FrameOptions { block_size_log2: 10, checksum, content_size };
        let frame = comp.compress_with(&input, opts);
        prop_assert_eq!(dec.decompress(&frame).unwrap(), input);
    }

    #[test]
    fn strings_roundtrip(dict in dictionary(), strings in prop::collection::vec(prop::collection::vec(alphabet_byte(), 0..40), 0..50)) {
        let comp = Compressor::new(&dict).unwrap();
        let dec = Decompressor::new(&dict).unwrap();
        let batch = comp.compress_strings(&strings);
        let batch = StringBatch::from_bytes(&batch.to_bytes()).unwrap();
        for (i, s) in strings.iter().enumerate() {
            let mut out = Vec::new();
            dec.decompress_string(&batch, i, &mut out).unwrap();
            prop_assert_eq!(&out, s);
        }
    }

    #[test]
    fn dictionary_bytes_roundtrip(dict in dictionary()) {
        let back = Dictionary::from_bytes(&dict.to_bytes()).unwrap();
        prop_assert_eq!(back.id(), dict.id());
        prop_assert_eq!(back, dict);
    }

    #[test]
    fn decoders_never_panic(dict in dictionary(), data in prop::collection::vec(any::<u8>(), 0..512), limit in 0usize..4096) {
        let dec = Decompressor::new(&dict).unwrap();
        let mut out = Vec::new();
        let _ = dec.decompress_block_into(&data, limit, &mut out);
        let _ = dec.decompress_codes_into(&data, &mut out);
        let _ = dec.decompress(&data);
        let _ = Dictionary::from_bytes(&data);
        let _ = StringBatch::from_bytes(&data);
        if let Ok(reference_out) = reference::decode_u8(&dict, &data) {
            let mut fast = Vec::new();
            prop_assert_eq!(dec.decompress_codes_into(&data, &mut fast).unwrap(), reference_out.len());
            prop_assert_eq!(fast, reference_out);
        } else {
            prop_assert!(dec.decompress_codes_into(&data, &mut Vec::new()).is_err());
        }
    }

    #[test]
    fn corrupted_frames_never_panic((dict, input) in dict_and_input(), flips in prop::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 1..4)) {
        let comp = Compressor::new(&dict).unwrap();
        let dec = Decompressor::new(&dict).unwrap();
        let mut frame = comp.compress_with(&input, FrameOptions { block_size_log2: 10, ..FrameOptions::default() });
        for (idx, x) in flips {
            let i = idx.index(frame.len());
            frame[i] ^= x;
        }
        if let Ok(out) = dec.decompress(&frame) {
            // With a checksum, a successful decode must be the original data.
            prop_assert_eq!(out, input);
        }
    }
}

#[test]
fn tail_lengths_and_all_escapes() {
    let dict = Dictionary::from_symbols(Scheme::U8, [&b"abcdefgh"[..], b"abc", b"ab", b"a"].iter().skip(1)).unwrap();
    let comp = Compressor::new(&dict).unwrap();
    let dec = Decompressor::new(&dict).unwrap();
    let text: Vec<u8> = b"abcabxaab\xff\x00abc".iter().copied().cycle().take(64).collect();
    for len in 0..=40 {
        let input = &text[..len];
        let mut codes = Vec::new();
        comp.compress_block_into(input, &mut codes);
        assert_eq!(codes, reference::encode_u8(&dict, input), "len {len}");
        let mut out = Vec::new();
        dec.decompress_block_into(&codes, len, &mut out).unwrap();
        assert_eq!(out, input);
    }
    // A dictionary without symbols escapes everything.
    let empty = Dictionary::from_symbols(Scheme::U8, std::iter::empty::<&[u8]>()).unwrap();
    let all: Vec<u8> = (0..=255).collect();
    let frame = Compressor::new(&empty).unwrap().compress(&all);
    assert_eq!(Decompressor::new(&empty).unwrap().decompress(&frame).unwrap(), all);
}
