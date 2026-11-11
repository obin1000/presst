//! `presst-eval`: compares presst with FSST, LZ4 and zstd on held-out data.
//!
//! Every input file is split by lines: the first half trains the dictionaries
//! (presst, FSST-once and zstd), the second half is the test data. Two
//! scenarios run on the test data:
//!
//! * blocks: the test data cut into independent blocks of each `--sizes` value;
//! * strings: every line compressed on its own (columnar random access).

use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use presst_bench::{BlockCodec, FsstOnce, FsstPerBlock, Lz4, Measurement, Presst, Zstd, measure};
use presst_train::{Corpus, Split, TrainOptions, train};

#[derive(Parser)]
#[command(about = "Compare presst with FSST, LZ4 and zstd on held-out data")]
struct Args {
    /// Line-based data files (for example the FSST dbtext corpus).
    #[arg(required = true)]
    files: Vec<PathBuf>,
    /// Block sizes in bytes for the blocks scenario.
    #[arg(long, value_delimiter = ',', default_value = "128,1024,16384,1048576")]
    sizes: Vec<usize>,
    /// Skip the per-string scenario.
    #[arg(long)]
    no_strings: bool,
    /// Use at most this many MiB of test data per file.
    #[arg(long, default_value_t = 4.0)]
    max_test_mib: f64,
    /// Minimum seconds per timing loop.
    #[arg(long, default_value_t = 0.2)]
    min_secs: f64,
    /// Append results as JSON lines to this file.
    #[arg(long)]
    json: Option<PathBuf>,
}

/// Name, block size (`None` for strings), inputs, and the codecs to run.
type Scenario<'a> = (String, Option<usize>, Vec<&'a [u8]>, &'a [Box<dyn BlockCodec>]);

#[derive(serde::Serialize)]
struct Row<'a> {
    dataset: &'a str,
    scenario: &'a str,
    block_size: Option<usize>,
    #[serde(flatten)]
    m: &'a Measurement,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut json = match &args.json {
        Some(p) => Some(
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
                .context("opening json output")?,
        ),
        None => None,
    };
    for file in &args.files {
        let name = file
            .file_name()
            .map_or_else(|| file.display().to_string(), |n| n.to_string_lossy().into_owned());
        let data = std::fs::read(file).with_context(|| format!("reading {}", file.display()))?;
        let lines: Vec<&[u8]> = data.split(|&b| b == b'\n').filter(|l| !l.is_empty()).collect();
        let (train_lines, test_lines) = lines.split_at(lines.len() / 2);
        let max_test = (args.max_test_mib * 1_048_576.0) as usize;
        let mut test_lines: Vec<&[u8]> = test_lines.to_vec();
        let mut total = 0;
        test_lines.retain(|l| {
            total += l.len() + 1;
            total <= max_test
        });
        let test: Vec<u8> = test_lines
            .iter()
            .flat_map(|l| l.iter().copied().chain(*b"\n"))
            .collect();

        eprintln!(
            "== {name}: training on {} lines, testing on {} KiB",
            train_lines.len(),
            test.len() / 1024
        );
        // Train for the way the data is used: text windows (with newlines)
        // for the blocks scenario, individual lines for the strings scenario.
        let train_text: Vec<u8> = train_lines
            .iter()
            .flat_map(|l| l.iter().copied().chain(*b"\n"))
            .collect();
        let mut text_corpus = Corpus::new();
        text_corpus.add(&train_text, Split::Windows(4096));
        let text_samples: Vec<&[u8]> = train_text.chunks(4096).collect();
        let line_corpus = Corpus::from_records(train_lines.iter().map(|l| l.to_vec()).collect());
        let block_codecs = codecs(&text_corpus, &text_samples, true)?;
        let string_codecs = if args.no_strings {
            Vec::new()
        } else {
            codecs(&line_corpus, train_lines, false)?
        };

        let mut scenarios: Vec<Scenario<'_>> = args
            .sizes
            .iter()
            .map(|&s| {
                (
                    format!("blocks {}", human(s)),
                    Some(s),
                    test.chunks(s).collect(),
                    block_codecs.as_slice(),
                )
            })
            .collect();
        if !args.no_strings {
            scenarios.push(("strings".into(), None, test_lines.clone(), string_codecs.as_slice()));
        }

        for (scenario, size, blocks, codecs) in &scenarios {
            println!("\n{name} / {scenario} ({} blocks)", blocks.len());
            println!(
                "  {:<28} {:>8} {:>12} {:>12}",
                "codec", "ratio", "comp MB/s", "decomp MB/s"
            );
            for codec in *codecs {
                let m = measure(codec.as_ref(), blocks, args.min_secs);
                println!(
                    "  {:<28} {:>8.3} {:>12.0} {:>12.0}",
                    m.codec, m.ratio, m.compress_mbps, m.decompress_mbps
                );
                if let Some(f) = &mut json {
                    let row = Row {
                        dataset: &name,
                        scenario: if size.is_some() { "blocks" } else { "strings" },
                        block_size: *size,
                        m: &m,
                    };
                    writeln!(f, "{}", serde_json::to_string(&row)?)?;
                }
            }
        }
    }
    Ok(())
}

/// The codecs to compare, with every trained component trained on `corpus`
/// (presst) or `samples` (FSST, zstd). Per-block FSST only makes sense for blocks.
fn codecs(corpus: &Corpus, samples: &[&[u8]], blocks: bool) -> Result<Vec<Box<dyn BlockCodec>>> {
    let ext = train(corpus, &TrainOptions::extended())?.dictionary;
    let fsst_mode = train(corpus, &TrainOptions::fsst())?.dictionary;
    let mut list: Vec<Box<dyn BlockCodec>> = vec![
        Box::new(Presst::new("presst (extended)", &ext)),
        Box::new(Presst::new("presst (fsst-trained)", &fsst_mode)),
        Box::new(FsstOnce::train(samples)),
    ];
    if blocks {
        list.push(Box::new(FsstPerBlock));
    }
    list.extend([
        Box::new(Lz4) as Box<dyn BlockCodec>,
        Box::new(Zstd::new(1)),
        Box::new(Zstd::new(3)),
    ]);
    for kib in [2usize, 110] {
        if let Some(z) = Zstd::with_trained_dict(3, samples, kib * 1024) {
            list.push(Box::new(z));
        }
    }
    Ok(list)
}

fn human(bytes: usize) -> String {
    match bytes {
        b if b >= 1 << 20 && b % (1 << 20) == 0 => format!("{} MiB", b >> 20),
        b if b >= 1 << 10 && b % (1 << 10) == 0 => format!("{} KiB", b >> 10),
        b => format!("{b} B"),
    }
}
