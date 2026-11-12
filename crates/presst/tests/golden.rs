//! Golden files: catch accidental changes to the binary formats.
//!
//! If a format change is intended, bump the format version, then regenerate:
//! `PRESST_BLESS=1 cargo test -p presst --test golden`.

use std::path::PathBuf;

use presst::{Compressor, Decompressor, Dictionary, FrameOptions, Metadata, Scheme, StringBatch};

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("golden")
}

fn check(name: &str, actual: &[u8]) {
    let path = golden_dir().join(name);
    if std::env::var_os("PRESST_BLESS").is_some() {
        std::fs::create_dir_all(golden_dir()).unwrap();
        std::fs::write(&path, actual).unwrap();
        return;
    }
    let expected = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e} (run with PRESST_BLESS=1)", path.display()));
    assert!(
        expected == actual,
        "{name} changed; if intended, bump the format version and re-bless"
    );
}

fn dictionary() -> Dictionary {
    Dictionary::from_symbols(
        Scheme::U8,
        [
            &b"the "[..],
            b"quick",
            b"brown ",
            b"fox",
            b" jumps",
            b"over",
            b"lazy",
            b"dog",
            b"e",
            b"o",
            b" ",
        ],
    )
    .unwrap()
    .with_metadata(Metadata {
        name: Some("golden".into()),
        trainer: Some("hand-written".into()),
        seed: Some(7),
        corpus_hash: Some(0xDEAD_BEEF),
    })
}

const TEXT: &[u8] = b"the quick brown fox jumps over the lazy dog; \xff\x00 the end";

#[test]
fn dictionary_format() {
    let dict = dictionary();
    check("dict-v1.pdict", &dict.to_bytes());
    let golden = std::fs::read(golden_dir().join("dict-v1.pdict")).unwrap();
    assert_eq!(Dictionary::from_bytes(&golden).unwrap(), dict);
    assert_eq!(
        format!("{}", dict.id()),
        "69410f036971bb16",
        "dictionary id is part of the format"
    );
}

#[test]
fn frame_format() {
    let dict = dictionary();
    let opts = FrameOptions {
        block_size_log2: 10,
        checksum: true,
        content_size: true,
        split_size_log2: None,
    };
    let frame = Compressor::new(&dict).unwrap().compress_with(TEXT, opts);
    check("frame-v1.prs", &frame);
    let golden = std::fs::read(golden_dir().join("frame-v1.prs")).unwrap();
    assert_eq!(Decompressor::new(&dict).unwrap().decompress(&golden).unwrap(), TEXT);
}

#[test]
fn frame_format_with_split_index() {
    let dict = dictionary();
    let opts = FrameOptions {
        block_size_log2: 10,
        checksum: true,
        content_size: true,
        split_size_log2: Some(8),
    };
    let frame = Compressor::new(&dict).unwrap().compress_with(TEXT, opts);
    check("frame-v2.prs", &frame);
    let golden = std::fs::read(golden_dir().join("frame-v2.prs")).unwrap();
    assert_eq!(Decompressor::new(&dict).unwrap().decompress(&golden).unwrap(), TEXT);
}

#[test]
fn batch_format() {
    let dict = dictionary();
    let batch = Compressor::new(&dict)
        .unwrap()
        .compress_strings(TEXT.split(|&b| b == b' '));
    check("batch-v1.prsc", &batch.to_bytes());
    let golden = std::fs::read(golden_dir().join("batch-v1.prsc")).unwrap();
    let parsed = StringBatch::from_bytes(&golden).unwrap();
    let (all, _) = Decompressor::new(&dict).unwrap().decompress_strings(&parsed).unwrap();
    assert_eq!(all, TEXT.iter().copied().filter(|&b| b != b' ').collect::<Vec<_>>());
}
