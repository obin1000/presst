//! GPU decode must be byte-identical to the CPU decoder and reject what it
//! rejects. Skipped (with a message) when no GPU backend is available.

use std::sync::OnceLock;

use presst::{Compressor, Decompressor, Dictionary, DictionaryBuilder, FrameOptions, Scheme};
use presst_cuda::{GpuContext, GpuError, TablePlacement};
use proptest::prelude::*;

fn gpu() -> Option<&'static GpuContext> {
    static GPU: OnceLock<Option<GpuContext>> = OnceLock::new();
    GPU.get_or_init(|| {
        if !presst_cuda::is_available() {
            eprintln!("presst-cuda: no GPU backend available, skipping GPU tests");
            return None;
        }
        Some(GpuContext::new(0).expect("open GPU 0"))
    })
    .as_ref()
}

fn alphabet_byte() -> impl Strategy<Value = u8> {
    prop_oneof![
        8 => prop::sample::select(b"abcde ".to_vec()),
        1 => any::<u8>(),
        1 => Just(0xFF),
    ]
}

fn dictionary() -> impl Strategy<Value = Dictionary> {
    prop::collection::vec(prop::collection::vec(alphabet_byte(), 1..=8), 0..300).prop_map(|syms| {
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

fn dict_and_input() -> impl Strategy<Value = (Dictionary, Vec<u8>)> {
    dictionary().prop_flat_map(|d| {
        let syms: Vec<Vec<u8>> = d.symbols().map(<[u8]>::to_vec).collect();
        let piece = if syms.is_empty() {
            prop::collection::vec(alphabet_byte(), 1..4).boxed()
        } else {
            prop_oneof![3 => prop::sample::select(syms), 1 => prop::collection::vec(alphabet_byte(), 1..4)].boxed()
        };
        (Just(d), prop::collection::vec(piece, 0..1500).prop_map(|p| p.concat()))
    })
}

fn options() -> impl Strategy<Value = FrameOptions> {
    (10u8..=12, prop::option::of(8u8..=11), any::<bool>(), any::<bool>()).prop_map(
        |(block, split, checksum, content_size)| FrameOptions {
            block_size_log2: block,
            split_size_log2: split,
            checksum,
            content_size,
        },
    )
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 300, ..ProptestConfig::default() })]

    #[test]
    fn gpu_decode_matches_cpu((dict, input) in dict_and_input(), opts in options()) {
        let Some(gpu) = gpu() else { return Ok(()) };
        let frame = Compressor::new(&dict).unwrap().compress_with(&input, opts);
        let table = gpu.upload(&dict).unwrap();
        for placement in [TablePlacement::Shared, TablePlacement::Global] {
            prop_assert_eq!(&gpu.decompress_frame(&table, &frame, placement).unwrap(), &input);
        }
    }

    #[test]
    fn gpu_rejects_what_cpu_rejects(
        (dict, input) in dict_and_input(),
        opts in options(),
        flips in prop::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 1..4),
    ) {
        let Some(gpu) = gpu() else { return Ok(()) };
        let mut frame = Compressor::new(&dict).unwrap().compress_with(&input, opts);
        for (idx, x) in flips {
            let i = idx.index(frame.len());
            frame[i] ^= x;
        }
        let cpu = Decompressor::new(&dict).unwrap().decompress(&frame);
        let table = gpu.upload(&dict).unwrap();
        let out = gpu.decompress_frame(&table, &frame, TablePlacement::Shared);
        match cpu {
            Ok(expected) => prop_assert_eq!(out.unwrap(), expected),
            Err(_) => prop_assert!(out.is_err()),
        }
    }
}

#[test]
fn large_frame_and_errors() {
    let Some(gpu) = gpu() else { return };
    let dict = Dictionary::from_symbols(Scheme::U8, [&b"lorem "[..], b"ipsum ", b"dolor ", b"sit ", b"amet"])
        .unwrap()
        .to_fsst_order();
    let input: Vec<u8> = b"lorem ipsum dolor sit amet, consectetur "
        .iter()
        .copied()
        .cycle()
        .take(5_000_000)
        .collect();
    let frame = Compressor::new(&dict).unwrap().compress_with(
        &input,
        FrameOptions {
            split_size_log2: Some(12),
            ..FrameOptions::default()
        },
    );
    let table = gpu.upload(&dict).unwrap();
    for placement in [TablePlacement::Shared, TablePlacement::Global] {
        assert_eq!(gpu.decompress_frame(&table, &frame, placement).unwrap(), input);
    }
    // The wrong dictionary is refused before any GPU work.
    let other = gpu
        .upload(&Dictionary::from_symbols(Scheme::U8, [&b"x"[..]]).unwrap())
        .unwrap();
    assert!(matches!(
        gpu.decompress_frame(&other, &frame, TablePlacement::Shared),
        Err(GpuError::Format(presst::Error::DictionaryMismatch { .. }))
    ));
}
