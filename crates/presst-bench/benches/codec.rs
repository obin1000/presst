//! Microbenchmarks of the presst kernels.
//!
//! Uses `corpora/data/dbtext/l_comment` when present (`cargo xtask corpus`),
//! otherwise synthetic text. Run with `cargo bench -p presst-bench`.

use std::fmt::Write as _;
use std::sync::OnceLock;

use divan::counter::BytesCount;
use divan::{Bencher, black_box};
use presst::{Compressor, Decompressor, Dictionary};
use presst_train::{Corpus, Split, TrainOptions, train};

fn main() {
    divan::main();
}

struct Setup {
    dict: Dictionary,
    test: Vec<u8>,
    lines: Vec<Vec<u8>>,
}

fn setup() -> &'static Setup {
    static SETUP: OnceLock<Setup> = OnceLock::new();
    SETUP.get_or_init(|| {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../corpora/data/dbtext/l_comment");
        let data = std::fs::read(path).unwrap_or_else(|_| {
            let mut text = String::new();
            for i in 0..40_000u32 {
                let _ = writeln!(text, "{i} slyly ironic requests boost carefully final deposits");
            }
            text.into_bytes()
        });
        let half = data.len() / 2;
        let (train_part, test) = data.split_at(half);
        let mut corpus = Corpus::new();
        corpus.add(train_part, Split::Windows(4096));
        let opts = TrainOptions {
            local_search_steps: 50,
            ..TrainOptions::extended()
        };
        let dict = train(&corpus, &opts).expect("train").dictionary;
        let lines = test
            .split(|&b| b == b'\n')
            .filter(|l| !l.is_empty())
            .map(<[u8]>::to_vec)
            .collect();
        Setup {
            dict,
            test: test[..test.len().min(1 << 20)].to_vec(),
            lines,
        }
    })
}

#[divan::bench]
fn compress_1mib(b: Bencher<'_, '_>) {
    let s = setup();
    let comp = Compressor::new(&s.dict).unwrap();
    let mut out = Vec::with_capacity(2 * s.test.len() + 16);
    b.counter(BytesCount::of_slice(&s.test)).bench_local(|| {
        out.clear();
        comp.compress_block_into(black_box(&s.test), &mut out)
    });
}

#[divan::bench]
fn decompress_1mib(b: Bencher<'_, '_>) {
    let s = setup();
    let mut codes = Vec::new();
    Compressor::new(&s.dict)
        .unwrap()
        .compress_block_into(&s.test, &mut codes);
    let dec = Decompressor::new(&s.dict).unwrap();
    let mut out = Vec::with_capacity(s.test.len() + 64);
    b.counter(BytesCount::of_slice(&s.test)).bench_local(|| {
        out.clear();
        dec.decompress_block_into(black_box(&codes), s.test.len(), &mut out)
            .unwrap();
    });
}

#[divan::bench]
fn compress_strings(b: Bencher<'_, '_>) {
    let s = setup();
    let comp = Compressor::new(&s.dict).unwrap();
    let bytes: usize = s.lines.iter().map(Vec::len).sum();
    b.counter(BytesCount::new(bytes))
        .bench_local(|| comp.compress_strings(black_box(&s.lines)));
}

#[divan::bench]
fn decompress_one_string(b: Bencher<'_, '_>) {
    let s = setup();
    let batch = Compressor::new(&s.dict).unwrap().compress_strings(&s.lines);
    let dec = Decompressor::new(&s.dict).unwrap();
    let mut out = Vec::with_capacity(256);
    let mut i = 0;
    b.bench_local(|| {
        out.clear();
        i = (i + 7919) % batch.len();
        dec.decompress_string(&batch, black_box(i), &mut out).unwrap();
    });
}

#[divan::bench]
fn prepare_tables(b: Bencher<'_, '_>) {
    let s = setup();
    b.bench_local(|| {
        (
            Compressor::new(black_box(&s.dict)).unwrap(),
            Decompressor::new(&s.dict).unwrap(),
        )
    });
}
