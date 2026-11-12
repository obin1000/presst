//! `cargo xtask data <dataset>`: download benchmark datasets and extract
//! string columns into `.vals` files with a train/test split.
//!
//! Layout: `corpora/data/columns/<dataset>/<table>/<column>.{train,test}.vals`
//! plus `meta.json` per table describing the split, and optionally
//! `<column>.test.time` with the seconds elapsed since the end of the
//! training period for every test value (for the aging test).

pub mod clickbench;
pub mod loghub;
pub mod publicbi;

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use md5::Md5;
use presst_bench::vals::{write_i64s, write_vals};
use sha2::{Digest, Sha256};

/// Maximum bytes of training and of test data per column.
pub const MAX_BYTES: usize = 64 << 20;
/// Columns with fewer values are skipped.
pub const MIN_VALUES: usize = 10_000;
/// Columns with a shorter average value are skipped (a database would
/// dictionary-encode them).
pub const MIN_AVG_LEN: f64 = 4.0;

pub fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

pub fn columns_dir() -> PathBuf {
    root().join("corpora").join("data").join("columns")
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// Verifies `data` against `md5:<hex>` or `sha256:<hex>`.
fn checksum_ok(data: &[u8], expected: &str) -> Result<bool> {
    Ok(match expected.split_once(':') {
        Some(("md5", h)) => hex(&Md5::digest(data)) == h,
        Some(("sha256", h)) => hex(&Sha256::digest(data)) == h,
        _ => bail!("unknown checksum {expected}"),
    })
}

/// Downloads every entry of `corpora/<dataset>.sources` that is missing or
/// corrupt and returns the local paths in manifest order.
pub fn fetch(dataset: &str) -> Result<Vec<PathBuf>> {
    let manifest = root().join("corpora").join(format!("{dataset}.sources"));
    let text = std::fs::read_to_string(&manifest).with_context(|| format!("reading {}", manifest.display()))?;
    let dir = root().join("corpora").join("download").join(dataset);
    std::fs::create_dir_all(&dir)?;
    let mut paths = Vec::new();
    for line in text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
    {
        let mut parts = line.split_whitespace();
        let (sum, url) = (
            parts.next().context("bad sources line")?,
            parts.next().context("bad sources line")?,
        );
        // Optional `bytes=<from>-<to>`: only that prefix is downloaded.
        let range = parts.next().and_then(|r| r.strip_prefix("bytes="));
        let name = url
            .rsplit('/')
            .next()
            .context("url without file name")?
            .split('?')
            .next()
            .unwrap_or_default();
        let path = dir.join(name);
        let ok = match std::fs::read(&path) {
            Ok(data) => checksum_ok(&data, sum)?,
            Err(_) => false,
        };
        if !ok {
            eprintln!("downloading {url}");
            let mut cmd = Command::new("curl");
            cmd.args(["-sSfL", "--retry", "3"]);
            if let Some(r) = range {
                cmd.args(["-r", r]);
            }
            let status = cmd.arg("-o").arg(&path).arg(url).status()?;
            if !status.success() {
                bail!("download failed: {url}");
            }
            if !checksum_ok(&std::fs::read(&path)?, sum)? {
                bail!("{name}: checksum does not match {sum}");
            }
        }
        paths.push(path);
    }
    Ok(paths)
}

/// Keeps at most `max_bytes` of `values` by taking every k-th value, so the
/// sample still spans the whole range (for example the whole time period).
/// Returns the kept indices.
pub fn spread_sample(values: &[Vec<u8>], max_bytes: usize) -> Vec<usize> {
    let total: usize = values.iter().map(|v| v.len() + 4).sum();
    if total <= max_bytes {
        return (0..values.len()).collect();
    }
    let step = total as f64 / max_bytes as f64;
    let mut out = Vec::new();
    let mut next = 0.0f64;
    let mut pos = 0usize;
    for (i, v) in values.iter().enumerate() {
        if pos as f64 >= next {
            out.push(i);
            next += step * (v.len() + 4) as f64;
        }
        pos += v.len() + 4;
    }
    out
}

/// Description of how a column was split, stored in `meta.json`.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Meta {
    pub dataset: String,
    pub table: String,
    /// Training level: "L1" (disjoint rows), "L2" (past -> future), "L3" (new table).
    pub level: String,
    pub split: String,
    /// Files the training values came from.
    pub train_sources: Vec<String>,
    /// Files the test values came from.
    pub test_sources: Vec<String>,
    pub columns: Vec<ColumnMeta>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ColumnMeta {
    pub name: String,
    pub train_values: usize,
    pub train_bytes: usize,
    pub test_values: usize,
    pub test_bytes: usize,
    /// NULL or empty values dropped.
    pub dropped: usize,
    pub has_time: bool,
}

/// One extracted column before writing.
pub struct Column {
    pub name: String,
    pub train: Vec<Vec<u8>>,
    pub test: Vec<Vec<u8>>,
    /// Seconds since the end of the training period, per test value.
    pub test_time: Option<Vec<i64>>,
    pub dropped: usize,
}

/// Applies the size limits and filters, writes the columns and `meta.json`.
/// Returns the number of columns written.
pub fn write_table(mut meta: Meta, columns: Vec<Column>) -> Result<usize> {
    let dir = columns_dir().join(&meta.dataset).join(&meta.table);
    std::fs::create_dir_all(&dir)?;
    for mut col in columns {
        let n = col.train.len() + col.test.len();
        let bytes: usize = col.train.iter().chain(&col.test).map(Vec::len).sum();
        if n < MIN_VALUES || (bytes as f64 / n.max(1) as f64) < MIN_AVG_LEN {
            eprintln!(
                "  skip {}/{}: {n} values, {:.1} bytes avg",
                meta.table,
                col.name,
                bytes as f64 / n.max(1) as f64
            );
            continue;
        }
        let keep = spread_sample(&col.train, MAX_BYTES);
        col.train = keep.into_iter().map(|i| std::mem::take(&mut col.train[i])).collect();
        let keep = spread_sample(&col.test, MAX_BYTES);
        if let Some(t) = &mut col.test_time {
            *t = keep.iter().map(|&i| t[i]).collect();
        }
        col.test = keep.into_iter().map(|i| std::mem::take(&mut col.test[i])).collect();

        write_vals(&dir.join(format!("{}.train.vals", col.name)), &col.train)?;
        write_vals(&dir.join(format!("{}.test.vals", col.name)), &col.test)?;
        if let Some(t) = &col.test_time {
            write_i64s(&dir.join(format!("{}.test.time", col.name)), t)?;
        }
        meta.columns.push(ColumnMeta {
            name: col.name.clone(),
            train_values: col.train.len(),
            train_bytes: col.train.iter().map(Vec::len).sum(),
            test_values: col.test.len(),
            test_bytes: col.test.iter().map(Vec::len).sum(),
            dropped: col.dropped,
            has_time: col.test_time.is_some(),
        });
        eprintln!(
            "  {}/{}: train {} values ({:.1} MB), test {} values ({:.1} MB)",
            meta.table,
            col.name,
            col.train.len(),
            meta.columns.last().expect("pushed").train_bytes as f64 / 1e6,
            col.test.len(),
            meta.columns.last().expect("pushed").test_bytes as f64 / 1e6
        );
    }
    let written = meta.columns.len();
    if written == 0 {
        std::fs::remove_dir_all(&dir).ok();
    } else {
        std::fs::write(dir.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;
    }
    Ok(written)
}

/// Reads a whole reader into memory.
pub fn read_all(mut r: impl Read) -> Result<Vec<u8>> {
    let mut v = Vec::new();
    r.read_to_end(&mut v)?;
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spread_sample_keeps_range() {
        let values: Vec<Vec<u8>> = (0..1000).map(|i: u32| i.to_le_bytes().to_vec()).collect();
        let keep = spread_sample(&values, 800);
        assert!(keep.len() >= 90 && keep.len() <= 110, "{}", keep.len());
        assert_eq!(keep[0], 0);
        assert!(*keep.last().unwrap() > 980);
        assert_eq!(spread_sample(&values, 1 << 20).len(), 1000);
    }

    #[test]
    fn checksums() {
        assert!(checksum_ok(b"abc", "md5:900150983cd24fb0d6963f7d28e17f72").unwrap());
        assert!(
            checksum_ok(
                b"abc",
                "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
            )
            .unwrap()
        );
        assert!(!checksum_ok(b"abd", "md5:900150983cd24fb0d6963f7d28e17f72").unwrap());
    }
}
