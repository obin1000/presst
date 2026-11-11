//! `presst-colbench`: database string column benchmark on real data.
//!
//! Reads columns prepared by `cargo xtask data` (`corpora/data/columns/<dataset>/<table>/`),
//! trains every trained codec on the column's training values only, and
//! measures ratio, compression speed, full-scan speed and point-lookup latency
//! on the test values. See `docs/BENCHMARKS.md` for the method.

use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use presst::Dictionary;
use presst_bench::column::{
    ColumnCodec, ColumnResult, FsstRowGroup, FsstShared, PerVector, PresstColumn, measure_column,
};
use presst_bench::vals::read_vals;
use presst_bench::{Lz4, Zstd};
use presst_train::{Corpus, TrainOptions, train};

#[derive(Parser)]
#[command(about = "Database string column benchmark on real data")]
struct Args {
    /// Directory with prepared columns.
    #[arg(long, default_value = "corpora/data/columns")]
    data: PathBuf,
    /// Only these datasets (for example `loghub`).
    #[arg(long, value_delimiter = ',')]
    datasets: Vec<String>,
    /// Only tables whose name contains one of these strings.
    #[arg(long, value_delimiter = ',')]
    tables: Vec<String>,
    /// Quick smoke run: at most 200k test values per column, 1 timed run.
    #[arg(long)]
    quick: bool,
    /// Timed runs per measurement (median is reported).
    #[arg(long, default_value_t = 5)]
    runs: usize,
    /// Point lookups per measurement.
    #[arg(long, default_value_t = 100_000)]
    lookups: usize,
    /// Append results as JSON lines to this file.
    #[arg(long)]
    json: Option<PathBuf>,
    /// Save trained presst dictionaries here (published with the results).
    #[arg(long)]
    save_dicts: Option<PathBuf>,
}

#[derive(serde::Deserialize)]
struct Meta {
    dataset: String,
    table: String,
    level: String,
    columns: Vec<MetaColumn>,
}

#[derive(serde::Deserialize)]
struct MetaColumn {
    name: String,
}

#[derive(serde::Serialize)]
struct Row<'a> {
    dataset: &'a str,
    table: &'a str,
    column: &'a str,
    level: &'a str,
    train_bytes: usize,
    dict_id: Option<String>,
    train_secs: f64,
    #[serde(flatten)]
    r: &'a ColumnResult,
}

#[allow(clippy::too_many_lines, reason = "split up when training levels are added")]
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
    if let Some(d) = &args.save_dicts {
        std::fs::create_dir_all(d)?;
    }
    for meta_path in find_tables(&args.data)? {
        let meta: Meta = serde_json::from_slice(&std::fs::read(&meta_path)?)?;
        if !args.datasets.is_empty() && !args.datasets.contains(&meta.dataset) {
            continue;
        }
        if !args.tables.is_empty() && !args.tables.iter().any(|t| meta.table.contains(t.as_str())) {
            continue;
        }
        let dir = meta_path.parent().expect("table dir");
        for col in &meta.columns {
            let train_vals = read_vals(&dir.join(format!("{}.train.vals", col.name)))?;
            let mut test_vals = read_vals(&dir.join(format!("{}.test.vals", col.name)))?;
            if args.quick {
                test_vals.truncate(200_000);
            }
            let train_refs: Vec<&[u8]> = train_vals.iter().map(Vec::as_slice).collect();
            let test_refs: Vec<&[u8]> = test_vals.iter().map(Vec::as_slice).collect();
            let train_bytes: usize = train_refs.iter().map(|v| v.len()).sum();
            let test_bytes: usize = test_refs.iter().map(|v| v.len()).sum();
            println!(
                "\n{}/{}/{} [{}]: train {:.1} MB, test {} values ({:.1} MB)",
                meta.dataset,
                meta.table,
                col.name,
                meta.level,
                train_bytes as f64 / 1e6,
                test_refs.len(),
                test_bytes as f64 / 1e6
            );

            // Every trained codec sees exactly the same training values.
            let mut codecs: Vec<(Box<dyn ColumnCodec>, Option<String>, f64)> = Vec::new();
            for (label, opts) in [
                ("presst", TrainOptions::extended()),
                ("presst (fsst-trained)", TrainOptions::fsst()),
            ] {
                let t = Instant::now();
                let dict = train_dict(&train_vals, &opts)?;
                let secs = t.elapsed().as_secs_f64();
                if let Some(d) = &args.save_dicts {
                    let name = format!(
                        "{}-{}-{}-{}.pdict",
                        meta.dataset,
                        meta.table,
                        col.name,
                        if label == "presst" { "extended" } else { "fsst" }
                    );
                    std::fs::write(d.join(name), dict.to_bytes())?;
                }
                codecs.push((
                    Box::new(PresstColumn::new(label, &dict)),
                    Some(dict.id().to_string()),
                    secs,
                ));
            }
            let t = Instant::now();
            codecs.push((
                Box::new(FsstShared::train(&spread(&train_refs, 100_000))),
                None,
                t.elapsed().as_secs_f64(),
            ));
            codecs.push((Box::new(FsstRowGroup), None, 0.0));
            codecs.push((Box::new(PerVector::new(Lz4)), None, 0.0));
            codecs.push((Box::new(PerVector::new(Zstd::new(3))), None, 0.0));
            let t = Instant::now();
            if let Some(z) = Zstd::with_trained_dict(3, &spread(&train_refs, 100_000), 110 * 1024) {
                codecs.push((Box::new(PerVector::new(z)), None, t.elapsed().as_secs_f64()));
            }

            println!(
                "  {:<28} {:>7} {:>9} {:>10} {:>10} {:>11}",
                "codec", "ratio", "shared", "comp MB/s", "scan MB/s", "lookup ns"
            );
            let runs = if args.quick { 1 } else { args.runs };
            for (codec, dict_id, train_secs) in &codecs {
                let r = measure_column(codec.as_ref(), &test_refs, runs, args.lookups);
                println!(
                    "  {:<28} {:>7.3} {:>9} {:>10.0} {:>10.0} {:>11.0}",
                    r.codec,
                    r.ratio,
                    human(r.shared_bytes),
                    r.compress_mbps,
                    r.scan_mbps,
                    r.lookup_ns
                );
                if let Some(f) = &mut json {
                    let row = Row {
                        dataset: &meta.dataset,
                        table: &meta.table,
                        column: &col.name,
                        level: &meta.level,
                        train_bytes,
                        dict_id: dict_id.clone(),
                        train_secs: *train_secs,
                        r: &r,
                    };
                    writeln!(f, "{}", serde_json::to_string(&row)?)?;
                }
            }
        }
    }
    Ok(())
}

fn train_dict(train_vals: &[Vec<u8>], opts: &TrainOptions) -> Result<Dictionary> {
    let corpus = Corpus::from_records(train_vals.to_vec());
    Ok(train(&corpus, opts)?.dictionary)
}

/// At most `n` values, evenly spread over `values`.
fn spread<'a>(values: &[&'a [u8]], n: usize) -> Vec<&'a [u8]> {
    if values.len() <= n {
        return values.to_vec();
    }
    (0..n).map(|i| values[i * values.len() / n]).collect()
}

fn find_tables(root: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))? {
            let p = entry?.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "meta.json") {
                out.push(p);
            }
        }
    }
    out.sort();
    Ok(out)
}

fn human(bytes: usize) -> String {
    match bytes {
        0 => "-".into(),
        b if b >= 1024 => format!("{:.1} KiB", b as f64 / 1024.0),
        b => format!("{b} B"),
    }
}
