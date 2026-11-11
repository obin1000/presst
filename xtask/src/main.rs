//! Project automation: `cargo xtask <command>`.
//!
//! * `corpus`: download the benchmark corpus listed in `corpora/*.sha256`
//!   into `corpora/data/` and verify every file.
//! * `report <results.jsonl>`: summarize `presst-eval` results as Markdown.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("corpus") => corpus(),
        Some("report") => report(Path::new(
            args.get(1).context("usage: cargo xtask report <results.jsonl>")?,
        )),
        _ => {
            eprintln!("usage: cargo xtask <corpus | report <results.jsonl>>");
            std::process::exit(2);
        }
    }
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

fn sha256(data: &[u8]) -> String {
    use std::fmt::Write as _;
    Sha256::digest(data).iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// Downloads every manifest entry that is missing or has the wrong hash.
fn corpus() -> Result<()> {
    let dir = root().join("corpora");
    for entry in std::fs::read_dir(&dir)? {
        let manifest = entry?.path();
        if manifest.extension().is_none_or(|e| e != "sha256") {
            continue;
        }
        let name = manifest.file_stem().expect("file name").to_string_lossy().into_owned();
        let out_dir = dir.join("data").join(&name);
        std::fs::create_dir_all(&out_dir)?;
        let text = std::fs::read_to_string(&manifest)?;
        let mut source = None;
        for line in text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
            let (a, b) = line.split_once(' ').context("bad manifest line")?;
            if a == "source" {
                source = Some(b.to_owned());
                continue;
            }
            let (hash, file) = (a, b.trim());
            let path = out_dir.join(file);
            if std::fs::read(&path).is_ok_and(|d| sha256(&d) == hash) {
                continue;
            }
            let url = format!("{}{file}", source.as_deref().context("manifest has no source line")?);
            eprintln!("downloading {url}");
            let status = Command::new("curl")
                .args(["-sSfL", "--retry", "3", "-o"])
                .arg(&path)
                .arg(&url)
                .status()?;
            if !status.success() {
                bail!("download failed: {url}");
            }
            let got = sha256(&std::fs::read(&path)?);
            if got != hash {
                bail!("{file}: sha256 {got} does not match the manifest");
            }
        }
        eprintln!("corpus {name}: ok ({})", out_dir.display());
    }
    Ok(())
}

#[derive(serde::Deserialize)]
struct Row {
    dataset: String,
    scenario: String,
    block_size: Option<usize>,
    codec: String,
    input_bytes: u64,
    compressed_bytes: u64,
    compress_mbps: f64,
    decompress_mbps: f64,
}

/// Input bytes, output bytes, compress seconds, decompress seconds, datasets.
type Totals = (u64, u64, f64, f64, usize);

/// Aggregates per scenario and codec over all datasets: total ratio
/// (sum of inputs / sum of outputs) and throughput over the total input
/// (total bytes / total time), so large and small datasets weigh by size.
fn report(path: &Path) -> Result<()> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    // (scenario order, label) -> codec -> (input, output, comp secs, decomp secs, datasets)
    let mut groups: BTreeMap<(usize, String), BTreeMap<String, Totals>> = BTreeMap::new();
    let mut datasets = std::collections::BTreeSet::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let r: Row = serde_json::from_str(line)?;
        datasets.insert(r.dataset.clone());
        let (order, label) = match (r.scenario.as_str(), r.block_size) {
            ("blocks", Some(s)) => (s, format!("{} blocks", human(s))),
            _ => (usize::MAX, "individual strings".to_owned()),
        };
        let e = groups
            .entry((order, label))
            .or_default()
            .entry(normalize(&r.codec))
            .or_default();
        let mb = r.input_bytes as f64 / 1e6;
        e.0 += r.input_bytes;
        e.1 += r.compressed_bytes;
        e.2 += mb / r.compress_mbps;
        e.3 += mb / r.decompress_mbps;
        e.4 += 1;
    }
    println!("Held-out evaluation on {} datasets, single thread.\n", datasets.len());
    for ((_, label), codecs) in &groups {
        println!("**{label}**\n");
        println!("| Codec | Ratio | Compress MB/s | Decompress MB/s |");
        println!("|---|---:|---:|---:|");
        for (codec, (input, output, cs, ds, _)) in codecs {
            let mb = *input as f64 / 1e6;
            println!(
                "| {codec} | {:.2} | {:.0} | {:.0} |",
                *input as f64 / (*output).max(1) as f64,
                mb / cs,
                mb / ds
            );
        }
        println!();
    }
    Ok(())
}

/// Older results named zstd dictionaries by their actual size, which varies
/// per dataset; group them by the requested maximum instead.
fn normalize(codec: &str) -> String {
    if let Some(rest) = codec.strip_prefix("zstd -3 + ") {
        if let Some(kib) = rest.strip_suffix(" KiB dict") {
            let max = if kib == "2" { 2 } else { 110 };
            return format!("zstd -3 + dict (max {max} KiB)");
        }
    }
    codec.to_owned()
}

fn human(bytes: usize) -> String {
    match bytes {
        b if b >= 1 << 20 && b % (1 << 20) == 0 => format!("{} MiB", b >> 20),
        b if b >= 1 << 10 && b % (1 << 10) == 0 => format!("{} KiB", b >> 10),
        b => format!("{b} B"),
    }
}
