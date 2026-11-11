//! ClickBench `hits`: real web analytics. All string columns of the selected
//! partitions are read, rows are ordered by `EventTime`, and the earliest half
//! trains while the latest half is test data (level L2, past -> future).

use std::collections::BTreeMap;
use std::fs::File;

use anyhow::{Context, Result};
use parquet::basic::Type as PhysicalType;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::Field;
use parquet::schema::types::Type;

use super::{Column, Meta, fetch, write_table};

pub fn run() -> Result<()> {
    let paths = fetch("clickbench")?;
    // column name -> (event time, value)
    let mut columns: BTreeMap<String, Vec<(i64, Vec<u8>)>> = BTreeMap::new();
    let mut dropped: BTreeMap<String, usize> = BTreeMap::new();
    let mut sources = Vec::new();
    for path in &paths {
        let name = path.file_name().expect("file name").to_string_lossy().into_owned();
        eprintln!("clickbench {name}");
        sources.push(name);
        let reader = SerializedFileReader::new(File::open(path)?)?;
        let schema = reader.metadata().file_metadata().schema_descr_ptr();
        let root = schema.root_schema();
        let fields: Vec<_> = root
            .get_fields()
            .iter()
            .filter(|f| {
                f.is_primitive() && (f.get_physical_type() == PhysicalType::BYTE_ARRAY || f.name() == "EventTime")
            })
            .cloned()
            .collect();
        let projection = Type::group_type_builder(root.name()).with_fields(fields).build()?;
        for row in reader.get_row_iter(Some(projection))? {
            let row = row?;
            let mut time = None;
            let mut values = Vec::new();
            for (name, field) in row.get_column_iter() {
                match field {
                    Field::Long(t) if name == "EventTime" => time = Some(*t),
                    Field::Int(t) if name == "EventTime" => time = Some(i64::from(*t)),
                    Field::TimestampMillis(t) if name == "EventTime" => time = Some(*t / 1000),
                    Field::TimestampMicros(t) if name == "EventTime" => time = Some(*t / 1_000_000),
                    Field::Bytes(b) => values.push((name, b.data().to_vec())),
                    Field::Str(s) => values.push((name, s.as_bytes().to_vec())),
                    Field::Null => values.push((name, Vec::new())),
                    _ => {}
                }
            }
            let time = time.context("row without EventTime")?;
            for (name, v) in values {
                if v.is_empty() {
                    *dropped.entry(name.clone()).or_default() += 1;
                } else {
                    columns.entry(name.clone()).or_default().push((time, v));
                }
            }
        }
    }

    let mut out = Vec::new();
    for (name, mut vals) in columns {
        // Stable sort keeps the original order for equal times.
        vals.sort_by_key(|(t, _)| *t);
        let split = vals.len() / 2;
        let train_end = vals.get(split.wrapping_sub(1)).map_or(0, |(t, _)| *t);
        let test: Vec<(i64, Vec<u8>)> = vals.split_off(split);
        let test_time = test.iter().map(|(t, _)| t - train_end).collect();
        out.push(Column {
            name,
            train: vals.into_iter().map(|(_, v)| v).collect(),
            test: test.into_iter().map(|(_, v)| v).collect(),
            test_time: Some(test_time),
            dropped: 0,
        });
    }
    for c in &mut out {
        c.dropped = dropped.get(&c.name).copied().unwrap_or(0);
    }
    let meta = Meta {
        dataset: "clickbench".into(),
        table: "hits".into(),
        level: "L2".into(),
        split: "rows ordered by EventTime: earliest half train, latest half test".into(),
        train_sources: sources.iter().map(|s| format!("{s}#EventTime<=median")).collect(),
        test_sources: sources.iter().map(|s| format!("{s}#EventTime>median")).collect(),
        columns: Vec::new(),
    };
    let n = write_table(meta, out)?;
    eprintln!("clickbench: {n} columns");
    Ok(())
}
