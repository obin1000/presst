//! Public BI benchmark: real string columns from 46 Tableau workbooks.
//!
//! Only the first 24 MiB of each bzip2 file is downloaded; bzip2 decompresses
//! as a stream, so that prefix yields complete rows. Workbooks with several
//! tables of the same schema train on table 1 and test on table 2 (level L3,
//! new table); single-table workbooks split their rows in half (level L1).
//! VARCHAR columns come from the workbook's schema files in the repository,
//! pinned to one commit. Per workbook, the 5 largest string columns that pass
//! the filters are kept, so no single workbook dominates the run time.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

use super::{Column, MIN_AVG_LEN, MIN_VALUES, Meta, fetch, root, write_table};

/// Repository commit the schemas are read from.
const SCHEMA_COMMIT: &str = "cf9909083bf9ed185cd7f5121a2881e1fea0e7a9";
/// String columns kept per workbook.
const COLUMNS_PER_WORKBOOK: usize = 5;

pub fn run() -> Result<()> {
    let paths = fetch("publicbi")?;
    // workbook -> tables (sorted by table number)
    let mut workbooks: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    for p in paths {
        let file = p.file_name().expect("file name").to_string_lossy().into_owned();
        let table = file.trim_end_matches(".csv.bz2");
        let wb = table.rsplit_once('_').map_or(table, |(w, _)| w).to_owned();
        workbooks.entry(wb).or_default().push(p);
    }
    let mut total = 0;
    for (wb, mut tables) in workbooks {
        tables.sort_by_key(|p| table_number(p));
        eprintln!("publicbi {wb} ({} tables)", tables.len());
        let first = table_name(&tables[0]);
        let schema = varchar_columns(&wb, &first)?;
        let t1 = read_table(&tables[0], &schema)?;
        let (level, split, train_src, test_src, mut cols) = if tables.len() >= 2 {
            let t2 = read_table(&tables[1], &schema)?;
            let cols: Vec<Column> = schema
                .varchar
                .iter()
                .enumerate()
                .map(|(i, name)| Column {
                    name: sanitize(name),
                    train: t1.values[i].clone(),
                    test: t2.values[i].clone(),
                    test_time: None,
                    dropped: t1.dropped[i] + t2.dropped[i],
                })
                .collect();
            let t2name = table_name(&tables[1]);
            (
                "L3",
                format!("new table: train on {first}, test on {t2name}"),
                first.clone(),
                t2name,
                cols,
            )
        } else {
            let cols: Vec<Column> = schema
                .varchar
                .iter()
                .enumerate()
                .map(|(i, name)| {
                    let mut train = t1.values[i].clone();
                    let test = train.split_off(train.len() / 2);
                    Column {
                        name: sanitize(name),
                        train,
                        test,
                        test_time: None,
                        dropped: t1.dropped[i],
                    }
                })
                .collect();
            (
                "L1",
                "row order: first half of the rows train, second half test".into(),
                format!("{first}#rows[first half]"),
                format!("{first}#rows[second half]"),
                cols,
            )
        };
        // Keep the largest string columns that pass the filters.
        cols.retain(|c| {
            let n = c.train.len() + c.test.len();
            let bytes: usize = c.train.iter().chain(&c.test).map(Vec::len).sum();
            n >= MIN_VALUES && bytes as f64 / n as f64 >= MIN_AVG_LEN && !c.test.is_empty() && !c.train.is_empty()
        });
        cols.sort_by_key(|c| std::cmp::Reverse(c.test.iter().map(Vec::len).sum::<usize>()));
        cols.truncate(COLUMNS_PER_WORKBOOK);
        if t1.skipped_rows > 0 {
            eprintln!("  {} malformed rows skipped in {first}", t1.skipped_rows);
        }
        let meta = Meta {
            dataset: "publicbi".into(),
            table: wb.clone(),
            level: level.into(),
            split,
            train_sources: vec![train_src],
            test_sources: vec![test_src],
            columns: Vec::new(),
        };
        total += write_table(meta, cols)?;
    }
    eprintln!("publicbi: {total} columns");
    Ok(())
}

fn table_name(p: &Path) -> String {
    p.file_name()
        .expect("file name")
        .to_string_lossy()
        .trim_end_matches(".csv.bz2")
        .to_owned()
}

fn table_number(p: &Path) -> u32 {
    table_name(p)
        .rsplit('_')
        .next()
        .and_then(|n| n.parse().ok())
        .unwrap_or(0)
}

/// File-name-safe column name.
fn sanitize(name: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    s.trim_matches('_').chars().take(60).collect()
}

struct Schema {
    /// Number of columns in a row.
    width: usize,
    /// Positions of the VARCHAR columns.
    positions: Vec<usize>,
    /// Names of the VARCHAR columns.
    varchar: Vec<String>,
}

/// Reads `CREATE TABLE` from the repository (cached) and finds the VARCHAR columns.
fn varchar_columns(wb: &str, table: &str) -> Result<Schema> {
    let dir = root().join("corpora").join("download").join("publicbi").join("schemas");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{table}.table.sql"));
    if !path.exists() {
        let url = format!(
            "https://raw.githubusercontent.com/cwida/public_bi_benchmark/{SCHEMA_COMMIT}/benchmark/{wb}/tables/{table}.table.sql"
        );
        let status = Command::new("curl")
            .args(["-sSfL", "--retry", "3", "-o"])
            .arg(&path)
            .arg(&url)
            .status()?;
        if !status.success() {
            bail!("download failed: {url}");
        }
    }
    let sql = std::fs::read_to_string(&path)?;
    let mut width = 0;
    let mut positions = Vec::new();
    let mut varchar = Vec::new();
    for line in sql.lines().map(str::trim).filter(|l| l.starts_with('"')) {
        let (name, rest) = line[1..].split_once('"').context("bad schema line")?;
        if rest.trim_start().to_ascii_lowercase().starts_with("varchar") {
            positions.push(width);
            varchar.push(name.to_owned());
        }
        width += 1;
    }
    Ok(Schema {
        width,
        positions,
        varchar,
    })
}

struct Table {
    /// Per VARCHAR column, the non-empty values in row order.
    values: Vec<Vec<Vec<u8>>>,
    /// Per VARCHAR column, empty (NULL) values.
    dropped: Vec<usize>,
    skipped_rows: usize,
}

/// Decompresses the bzip2 prefix (stopping at the truncated end) and splits
/// `|`-separated rows into the VARCHAR columns.
fn read_table(path: &Path, schema: &Schema) -> Result<Table> {
    let mut decoder = bzip2::read::MultiBzDecoder::new(std::fs::File::open(path)?);
    let mut data = Vec::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        match decoder.read(&mut buf) {
            Ok(0) | Err(_) => break, // end of the downloaded prefix
            Ok(n) => data.extend_from_slice(&buf[..n]),
        }
    }
    // Drop the last, possibly incomplete row.
    if let Some(end) = data.iter().rposition(|&b| b == b'\n') {
        data.truncate(end);
    }
    let mut values = vec![Vec::new(); schema.varchar.len()];
    let mut dropped = vec![0; schema.varchar.len()];
    let mut skipped_rows = 0;
    for row in data.split(|&b| b == b'\n') {
        let row = row.strip_suffix(b"\r").unwrap_or(row);
        let fields: Vec<&[u8]> = row.split(|&b| b == b'|').collect();
        if fields.len() != schema.width {
            skipped_rows += 1;
            continue;
        }
        for (i, &pos) in schema.positions.iter().enumerate() {
            let v = fields[pos];
            if v.is_empty() || v == b"null" {
                dropped[i] += 1;
            } else {
                values[i].push(v.to_vec());
            }
        }
    }
    Ok(Table {
        values,
        dropped,
        skipped_rows,
    })
}
