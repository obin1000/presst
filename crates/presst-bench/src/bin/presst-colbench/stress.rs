//! Stress tests: where does pretraining break down?
//!
//! * `aging`: ratio by how long after the training period the data was produced.
//! * `matrix`: a column's dictionary applied to other columns.
//! * `sweep`: dictionaries trained on growing amounts of training data.
//! * `mix`: generic (L5) dictionaries trained on different mixes of datasets.
//!
//! All results are compression ratios on the test values; every trained codec
//! sees the same training data.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;
use presst_bench::column::{FsstRowGroup, PresstColumn, measure_ratio};
use presst_bench::vals::{read_i64s, read_vals};
use presst_train::{Corpus, TrainOptions, train};

use super::{Ctx, Origin, Source, Target, concat_sources, ratio_rows, train_all};

/// Age buckets in seconds after the end of the training period.
const AGE_BUCKETS: [(&str, i64, i64); 6] = [
    ("<1h", i64::MIN + 1, 3600),
    ("1h-1d", 3600, 86_400),
    ("1d-7d", 86_400, 7 * 86_400),
    ("7d-30d", 7 * 86_400, 30 * 86_400),
    ("30d-180d", 30 * 86_400, 180 * 86_400),
    (">180d", 180 * 86_400, i64::MAX),
];

/// Minimum values in an age bucket for it to be reported.
const MIN_BUCKET_VALUES: usize = 2000;

/// Columns used by the matrix and sweep: the largest columns per dataset.
pub(super) fn selected<'a>(
    ctx: &Ctx<'_>,
    tables: &'a [(std::path::PathBuf, super::Meta)],
) -> Vec<(&'a Path, Target<'a>, &'a str)> {
    let mut out = Vec::new();
    for (path, m) in tables {
        for c in &m.columns {
            if ctx
                .variance
                .contains(&(m.dataset.clone(), m.table.clone(), c.name.clone()))
            {
                let t = Target {
                    dataset: &m.dataset,
                    table: &m.table,
                    column: &c.name,
                };
                out.push((path.parent().expect("dir"), t, m.level.as_str()));
            }
        }
    }
    out
}

/// Ratio per age bucket, for columns with test timestamps.
pub(super) fn aging(ctx: &mut Ctx<'_>, t: &Target<'_>, level: &str, dir: &Path) -> Result<()> {
    let time_path = dir.join(format!("{}.test.time", t.column));
    let Ok(times) = read_i64s(&time_path) else {
        return Ok(());
    };
    let train_vals = read_vals(&dir.join(format!("{}.train.vals", t.column)))?;
    let test_vals = read_vals(&dir.join(format!("{}.test.vals", t.column)))?;
    let train_refs: Vec<&[u8]> = train_vals.iter().map(Vec::as_slice).collect();
    let native = train_all(&train_refs, &TrainOptions::extended())?;
    println!("\naging {}/{}/{}", t.dataset, t.table, t.column);
    for (name, lo, hi) in AGE_BUCKETS {
        let bucket: Vec<&[u8]> = test_vals
            .iter()
            .zip(&times)
            .filter(|&(_, &a)| a != i64::MIN && a >= lo && a < hi)
            .map(|(v, _)| v.as_slice())
            .collect();
        if bucket.len() < MIN_BUCKET_VALUES {
            continue;
        }
        let variant = format!("age:{name}");
        ratio_rows(&mut ctx.out, t, level, &variant, &native, &bucket)?;
        let o = Origin {
            level: "self",
            variant: &variant,
            trained: None,
            is_presst: false,
        };
        ctx.out.row(t, o, &measure_ratio(&FsstRowGroup, &bucket))?;
    }
    Ok(())
}

/// Every selected column's dictionary applied to every selected column.
pub(super) fn matrix(ctx: &mut Ctx<'_>, cols: &[(&Path, Target<'_>, &str)]) -> Result<()> {
    let mut dicts = Vec::new();
    for (dir, t, _) in cols {
        let train_vals = read_vals(&dir.join(format!("{}.train.vals", t.column)))?;
        let refs: Vec<&[u8]> = train_vals.iter().map(Vec::as_slice).collect();
        let corpus = Corpus::from_records(refs.iter().map(|v| v.to_vec()).collect());
        dicts.push(train(&corpus, &TrainOptions::extended())?.dictionary);
    }
    for (dir, target, level) in cols {
        let test_vals = read_vals(&dir.join(format!("{}.test.vals", target.column)))?;
        let test: Vec<&[u8]> = test_vals.iter().map(Vec::as_slice).collect();
        println!("\nmatrix -> {}/{}/{}", target.dataset, target.table, target.column);
        for ((_, from, _), dict) in cols.iter().zip(&dicts) {
            let variant = format!("from:{}/{}/{}", from.dataset, from.table, from.column);
            let r = measure_ratio(&PresstColumn::new("presst", dict), &test);
            ctx.out.row(
                target,
                Origin {
                    level,
                    variant: &variant,
                    trained: None,
                    is_presst: true,
                },
                &r,
            )?;
        }
    }
    Ok(())
}

/// Training-size sweep: the first N bytes of the training data (in time or
/// row order), for presst, FSST with a shared table and zstd with a dictionary.
pub(super) fn sweep(ctx: &mut Ctx<'_>, cols: &[(&Path, Target<'_>, &str)]) -> Result<()> {
    const SIZES: [usize; 6] = [16 << 10, 256 << 10, 1 << 20, 4 << 20, 16 << 20, 64 << 20];
    for (dir, t, level) in cols {
        let train_vals = read_vals(&dir.join(format!("{}.train.vals", t.column)))?;
        let test_vals = read_vals(&dir.join(format!("{}.test.vals", t.column)))?;
        let test: Vec<&[u8]> = test_vals.iter().map(Vec::as_slice).collect();
        let total: usize = train_vals.iter().map(Vec::len).sum();
        println!("\nsweep {}/{}/{}", t.dataset, t.table, t.column);
        for size in SIZES {
            if size > total * 2 {
                break;
            }
            let mut used = 0;
            let prefix: Vec<&[u8]> = train_vals
                .iter()
                .take_while(|v| {
                    used += v.len();
                    used <= size
                })
                .map(Vec::as_slice)
                .collect();
            let opts = TrainOptions {
                sample_bytes: size,
                ..TrainOptions::extended()
            };
            let trained = train_all(&prefix, &opts)?;
            ratio_rows(&mut ctx.out, t, level, &format!("size:{size}"), &trained, &test)?;
        }
    }
    Ok(())
}

/// Generic dictionaries trained on different mixes of the other datasets.
pub(super) fn mix(ctx: &mut Ctx<'_>, tables: &[(std::path::PathBuf, super::Meta)]) -> Result<()> {
    let datasets: Vec<String> = {
        let mut d: Vec<String> = tables.iter().map(|(_, m)| m.dataset.clone()).collect();
        d.sort();
        d.dedup();
        d
    };
    for target in &datasets {
        // Each mix leaves the target dataset out; variants drop one more source.
        let others: Vec<&str> = ctx
            .pool
            .iter()
            .map(|s| s.dataset.as_str())
            .filter(|d| d != target)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let mut mixes: BTreeMap<String, Vec<&Source>> = BTreeMap::new();
        mixes.insert(
            "all-others".into(),
            ctx.pool.iter().filter(|s| &s.dataset != target).collect(),
        );
        for drop in &others {
            mixes.insert(
                format!("without-{drop}"),
                ctx.pool
                    .iter()
                    .filter(|s| &s.dataset != target && s.dataset != *drop)
                    .collect(),
            );
        }
        let mut trained = Vec::new();
        for (name, sources) in &mixes {
            if sources.is_empty() {
                continue;
            }
            let dict = train(
                &Corpus::from_records(concat_sources(sources).iter().map(|v| v.to_vec()).collect()),
                &TrainOptions::extended(),
            )?
            .dictionary;
            trained.push((name.clone(), dict));
        }
        for (path, m) in tables.iter().filter(|(_, m)| &m.dataset == target) {
            for c in &m.columns {
                let test_vals = read_vals(&path.parent().expect("dir").join(format!("{}.test.vals", c.name)))?;
                let test: Vec<&[u8]> = test_vals.iter().map(Vec::as_slice).collect();
                let t = Target {
                    dataset: &m.dataset,
                    table: &m.table,
                    column: &c.name,
                };
                for (name, dict) in &trained {
                    let variant = format!("mix:{name}");
                    let r = measure_ratio(&PresstColumn::new("presst", dict), &test);
                    ctx.out.row(
                        &t,
                        Origin {
                            level: "L5",
                            variant: &variant,
                            trained: None,
                            is_presst: true,
                        },
                        &r,
                    )?;
                }
            }
        }
    }
    Ok(())
}
