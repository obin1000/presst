//! `cargo xtask e2e-dicts`: per-column dictionaries for the end-to-end
//! benchmarks (query engines reading presst-compressed string columns).
//!
//! Reads Parquet files, trains one dictionary per string column and writes
//! `<out>/<id>.pdict` plus `<out>/manifest.json`:
//!
//! ```json
//! {"format": 1, "columns": {"<table>.<column>": "<16 hex digit id>"}, "training": {...}}
//! ```
//!
//! `columns` is the contract with the engine integrations (Vortex, DuckDB);
//! `training` records what each dictionary was trained on. Dictionaries are
//! stored in FSST code order (`Dictionary::to_fsst_order`), so engines with
//! FSST kernels can work on the codes directly. Every string column with at
//! least `--min-values` non-empty values gets one; the engine still decides
//! per column whether to use it. An existing manifest in `<out>` is
//! extended, so several runs can share one directory.
//!
//! The table name is the file stem without a trailing `_<number>` (so
//! `lineitem_0.parquet` and `lineitem_1.parquet` are both `lineitem`), or the
//! value of `--table`.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use parquet::basic::{ConvertedType, LogicalType, Type as PhysicalType};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::Field;
use parquet::schema::types::Type;
use presst_train::{Corpus, TrainOptions};

use crate::data::spread_sample;

const USAGE: &str = "usage: cargo xtask e2e-dicts -o <dir> [--table <name>] [--min-values <n>] \
                     [--max-bytes <n>] [--fsst] <file.parquet | dir>...";

/// Manifest format version.
const FORMAT: u32 = 1;

#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Manifest {
    pub format: u32,
    /// `table.column` -> dictionary id (16 lowercase hex digits).
    pub columns: BTreeMap<String, String>,
    /// `table.column` -> what the dictionary was trained on.
    #[serde(default)]
    pub training: BTreeMap<String, Training>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Training {
    /// Parquet files the values came from (file names only).
    pub sources: Vec<String>,
    /// Non-empty values in the sources.
    pub values: usize,
    /// Bytes of those values.
    pub bytes: usize,
    /// Bytes actually used for training (evenly spread sample).
    pub sample_bytes: usize,
    /// Validation compression ratio reported by the trainer.
    pub validation_ratio: f64,
    /// Trainer mode: "extended" or "fsst".
    pub mode: String,
    pub train_seconds: f64,
}

struct Args {
    out: PathBuf,
    table: Option<String>,
    min_values: usize,
    max_bytes: usize,
    fsst: bool,
    inputs: Vec<PathBuf>,
}

fn parse(args: &[String]) -> Result<Args> {
    let mut out = None;
    let mut a = Args {
        out: PathBuf::new(),
        table: None,
        min_values: 1000,
        max_bytes: crate::data::MAX_BYTES,
        fsst: false,
        inputs: Vec::new(),
    };
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = || it.next().with_context(|| format!("{arg} needs a value\n{USAGE}"));
        match arg.as_str() {
            "-o" | "--out" => out = Some(PathBuf::from(value()?)),
            "--table" => a.table = Some(value()?.clone()),
            "--min-values" => a.min_values = value()?.parse()?,
            "--max-bytes" => a.max_bytes = value()?.parse()?,
            "--fsst" => a.fsst = true,
            s if s.starts_with('-') => bail!("unknown option {s}\n{USAGE}"),
            s => a.inputs.push(PathBuf::from(s)),
        }
    }
    a.out = out.context(USAGE)?;
    if a.inputs.is_empty() {
        bail!(USAGE);
    }
    Ok(a)
}

/// All `.parquet` files named by `inputs` (directories are searched recursively), sorted.
fn parquet_files(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
        for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
            let path = entry?.path();
            if path.is_dir() {
                walk(&path, out)?;
            } else if path.extension().is_some_and(|e| e == "parquet") {
                out.push(path);
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    for input in inputs {
        if input.is_dir() {
            walk(input, &mut files)?;
        } else {
            files.push(input.clone());
        }
    }
    files.sort();
    if files.is_empty() {
        bail!("no .parquet files found");
    }
    Ok(files)
}

/// `lineitem_3` -> `lineitem`; `hits` -> `hits`.
pub fn table_name(path: &Path) -> String {
    let stem = path.file_stem().expect("file name").to_string_lossy();
    match stem.rsplit_once('_') {
        Some((base, n)) if !base.is_empty() && !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) => {
            base.to_owned()
        }
        _ => stem.into_owned(),
    }
}

/// Whether a Parquet leaf holds strings (or untyped binary), not decimals or other types.
fn is_string(field: &Type) -> bool {
    if !field.is_primitive() || field.get_physical_type() != PhysicalType::BYTE_ARRAY {
        return false;
    }
    let info = field.get_basic_info();
    !matches!(info.logical_type_ref(), Some(LogicalType::Decimal { .. }))
        && info.converted_type() != ConvertedType::DECIMAL
}

/// table -> column -> (values, source files).
type Values = BTreeMap<String, BTreeMap<String, (Vec<Vec<u8>>, Vec<String>)>>;

fn read_values(files: &[PathBuf], table_override: Option<&str>) -> Result<Values> {
    let mut out: Values = BTreeMap::new();
    for path in files {
        let table = table_override.map_or_else(|| table_name(path), str::to_owned);
        let file_name = path.file_name().expect("file name").to_string_lossy().into_owned();
        eprintln!("reading {} (table {table})", path.display());
        let reader = SerializedFileReader::new(File::open(path)?)?;
        let schema = reader.metadata().file_metadata().schema_descr_ptr();
        let root = schema.root_schema();
        let fields: Vec<_> = root.get_fields().iter().filter(|f| is_string(f)).cloned().collect();
        if fields.is_empty() {
            continue;
        }
        let columns = out.entry(table).or_default();
        for f in &fields {
            let (_, sources) = columns.entry(f.name().to_owned()).or_default();
            sources.push(file_name.clone());
        }
        let projection = Type::group_type_builder(root.name()).with_fields(fields).build()?;
        for row in reader.get_row_iter(Some(projection))? {
            for (name, field) in row?.get_column_iter() {
                let v = match field {
                    Field::Bytes(b) => b.data(),
                    Field::Str(s) => s.as_bytes(),
                    _ => continue,
                };
                if !v.is_empty() {
                    columns.get_mut(name).expect("projected column").0.push(v.to_vec());
                }
            }
        }
    }
    Ok(out)
}

pub fn run(args: &[String]) -> Result<()> {
    let args = parse(args)?;
    let files = parquet_files(&args.inputs)?;
    std::fs::create_dir_all(&args.out)?;
    let manifest_path = args.out.join("manifest.json");
    let mut manifest: Manifest = if manifest_path.exists() {
        serde_json::from_str(&std::fs::read_to_string(&manifest_path)?)
            .with_context(|| format!("reading {}", manifest_path.display()))?
    } else {
        Manifest {
            format: FORMAT,
            ..Manifest::default()
        }
    };
    if manifest.format != FORMAT {
        bail!(
            "{} has format {}, expected {FORMAT}",
            manifest_path.display(),
            manifest.format
        );
    }

    let values = read_values(&files, args.table.as_deref())?;
    for (table, columns) in values {
        for (column, (vals, sources)) in columns {
            let key = format!("{table}.{column}");
            let bytes: usize = vals.iter().map(Vec::len).sum();
            if vals.len() < args.min_values {
                eprintln!(
                    "  skip {key}: {} values, {:.1} bytes avg",
                    vals.len(),
                    bytes as f64 / vals.len().max(1) as f64
                );
                continue;
            }
            let keep = spread_sample(&vals, args.max_bytes);
            let sample: Vec<Vec<u8>> = keep.into_iter().map(|i| vals[i].clone()).collect();
            let sample_bytes = sample.iter().map(Vec::len).sum();
            let mut opts = if args.fsst {
                TrainOptions::fsst()
            } else {
                TrainOptions::extended()
            };
            opts.name = Some(key.clone());
            let start = Instant::now();
            let report =
                presst_train::train(&Corpus::from_records(sample), &opts).with_context(|| format!("training {key}"))?;
            let train_seconds = start.elapsed().as_secs_f64();
            // FSST code order, so engines with FSST kernels can use the codes as is.
            let dict = report.dictionary.to_fsst_order();
            let id = dict.id().to_string();
            std::fs::write(args.out.join(format!("{id}.pdict")), dict.to_bytes())?;
            eprintln!(
                "  {key}: {} values ({:.1} MB), ratio {:.2}, {train_seconds:.1} s -> {id}",
                vals.len(),
                bytes as f64 / 1e6,
                report.validation_ratio
            );
            manifest.columns.insert(key.clone(), id);
            manifest.training.insert(
                key,
                Training {
                    sources,
                    values: vals.len(),
                    bytes,
                    sample_bytes,
                    validation_ratio: report.validation_ratio,
                    mode: if args.fsst { "fsst" } else { "extended" }.into(),
                    train_seconds,
                },
            );
        }
    }
    std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)? + "\n")?;
    eprintln!("{} dictionaries in {}", manifest.columns.len(), manifest_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_names() {
        assert_eq!(table_name(Path::new("a/lineitem_0.parquet")), "lineitem");
        assert_eq!(table_name(Path::new("lineitem_12.parquet")), "lineitem");
        assert_eq!(table_name(Path::new("hits.parquet")), "hits");
        assert_eq!(table_name(Path::new("store_sales.parquet")), "store_sales");
        assert_eq!(table_name(Path::new("_1.parquet")), "_1");
    }

    #[test]
    fn manifest_roundtrip() {
        let mut m = Manifest {
            format: FORMAT,
            ..Manifest::default()
        };
        m.columns.insert("t.c".into(), "0123456789abcdef".into());
        let text = serde_json::to_string(&m).unwrap();
        let back: Manifest = serde_json::from_str(&text).unwrap();
        assert_eq!(back.columns, m.columns);
        // `training` is optional for readers that only need the mapping.
        let minimal: Manifest = serde_json::from_str(r#"{"format":1,"columns":{"t.c":"00"}}"#).unwrap();
        assert_eq!(minimal.columns.len(), 1);
    }
}
