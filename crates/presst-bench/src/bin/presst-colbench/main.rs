//! `presst-colbench`: database string column benchmark on real data.
//!
//! Reads columns prepared by `cargo xtask data` (`corpora/data/columns/<dataset>/<table>/`)
//! and, per column, measures every codec on the test values. Trained codecs
//! (presst, FSST with a shared table, zstd with a dictionary) are trained on
//! exactly the same data at several *training levels*:
//!
//! | Level | Training data |
//! |---|---|
//! | L0 | the test values themselves (oracle, upper bound) |
//! | L1/L2/L3 | the column's own training split (disjoint rows / past / other table) |
//! | L4 | same kind of data from other sources, never this table |
//! | L5 | a generic mix of the *other* datasets, never this dataset |
//!
//! The native level (L1–L3) is fully timed; L0, L4 and L5 report the ratio.
//! Also measured: a test set without values that occur in the training data
//! (duplicate control), and presst's spread over training subsets and seeds.
//! `--stress` runs the stress tests in `stress.rs` instead.

mod stress;

use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use presst::Dictionary;
use presst_bench::column::{
    ColumnCodec, ColumnResult, Encoded, FsstRowGroup, FsstShared, PerVector, PresstColumn, SharedBytes, measure_column,
    measure_ratio,
};
use presst_bench::vals::read_vals;
use presst_bench::{BlockCodec, Lz4, Zstd};
use presst_train::{Corpus, TrainOptions, train};

#[derive(Parser)]
#[command(about = "Database string column benchmark on real data")]
struct Args {
    /// Directory with prepared columns.
    #[arg(long, default_value = "corpora/data/columns")]
    data: PathBuf,
    /// dbtext corpus, used as extra training material for L4/L5.
    #[arg(long, default_value = "corpora/data/dbtext")]
    dbtext: PathBuf,
    /// Only these datasets (for example `loghub`).
    #[arg(long, value_delimiter = ',')]
    datasets: Vec<String>,
    /// Only tables whose name contains one of these strings.
    #[arg(long, value_delimiter = ',')]
    tables: Vec<String>,
    /// Quick smoke run: at most 200k test values per column, 1 timed run, no extra levels.
    #[arg(long)]
    quick: bool,
    /// Timed runs per measurement (median is reported).
    #[arg(long, default_value_t = 3)]
    runs: usize,
    /// Point lookups per measurement.
    #[arg(long, default_value_t = 100_000)]
    lookups: usize,
    /// Columns per dataset (the largest) that get the seed/subset variance check.
    #[arg(long, default_value_t = 4)]
    variance: usize,
    /// Append results as JSON lines to this file.
    #[arg(long)]
    json: Option<PathBuf>,
    /// Save every trained presst dictionary here (published with the results).
    #[arg(long)]
    save_dicts: Option<PathBuf>,
    /// Run stress tests instead of the main benchmark: aging, matrix, sweep, mix.
    #[arg(long, value_delimiter = ',')]
    stress: Vec<String>,
}

#[derive(serde::Deserialize)]
struct Meta {
    dataset: String,
    table: String,
    level: String,
    train_sources: Vec<String>,
    test_sources: Vec<String>,
    columns: Vec<MetaColumn>,
}

#[derive(serde::Deserialize)]
struct MetaColumn {
    name: String,
    test_bytes: usize,
}

/// Kind of data, for choosing L4 training material.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Kind {
    Url,
    Text,
    Log,
    Other,
}

fn kind_of(dataset: &str, column: &str) -> Kind {
    let c = column.to_ascii_lowercase();
    match dataset {
        "loghub" => Kind::Log,
        "clickbench" => match column {
            "URL" | "Referer" | "OriginalURL" => Kind::Url,
            "Title" | "SearchPhrase" => Kind::Text,
            _ => Kind::Other,
        },
        "dbtext" => match column {
            "urls" | "urls2" => Kind::Url,
            "wiki" | "wikipedia" | "hamlet" | "faust" | "movies" | "l_comment" | "ps_comment" => Kind::Text,
            _ => Kind::Other,
        },
        _ if ["url", "link", "website", "http", "href"].iter().any(|k| c.contains(k)) => Kind::Url,
        _ if ["description", "title", "comment", "text", "desc"]
            .iter()
            .any(|k| c.contains(k)) =>
        {
            Kind::Text
        }
        _ => Kind::Other,
    }
}

/// A sample of one column's training data, for building L4/L5 training sets.
struct Source {
    dataset: String,
    table: String,
    kind: Kind,
    sample: Vec<Vec<u8>>,
}

/// Training bytes taken from each source column for L4/L5.
const POOL_BYTES: usize = 512 << 10;

/// The column being measured.
struct Target<'a> {
    dataset: &'a str,
    table: &'a str,
    column: &'a str,
}

/// L4 training set: same kind of data from other tables. Loghub: the other
/// systems (leave one out); Public BI: the other workbooks; ClickBench: URLs or
/// text from the other datasets. Returns `None` if there is no such material.
fn l4_sources<'p>(t: &Target<'_>, pool: &'p [Source]) -> Option<(String, Vec<&'p Source>)> {
    let kind = kind_of(t.dataset, t.column);
    let (desc, picked): (String, Vec<&Source>) = match t.dataset {
        "loghub" | "publicbi" => (
            format!("other {} tables", t.dataset),
            pool.iter()
                .filter(|s| s.dataset == t.dataset && s.table != t.table)
                .collect(),
        ),
        _ if kind != Kind::Other => (
            format!("{kind:?} columns of other datasets"),
            pool.iter()
                .filter(|s| s.dataset != t.dataset && s.kind == kind)
                .collect(),
        ),
        _ => return None,
    };
    (!picked.is_empty()).then_some((desc, picked))
}

/// L5 training set: everything from the other datasets.
fn l5_sources<'p>(t: &Target<'_>, pool: &'p [Source]) -> Vec<&'p Source> {
    pool.iter().filter(|s| s.dataset != t.dataset).collect()
}

/// Trained codecs that all saw the same training values.
struct Trained {
    dict: Dictionary,
    fsst: FsstShared,
    zstd: Option<Zstd>,
    train_bytes: usize,
    secs: f64,
}

fn train_all(values: &[&[u8]], opts: &TrainOptions) -> Result<Trained> {
    let t = Instant::now();
    let corpus = Corpus::from_records(values.iter().map(|v| v.to_vec()).collect());
    let dict = train(&corpus, opts)?.dictionary;
    let secs = t.elapsed().as_secs_f64();
    let samples = spread(values, 100_000);
    Ok(Trained {
        dict,
        fsst: FsstShared::train(&samples),
        zstd: Zstd::with_trained_dict(3, &samples, 110 * 1024),
        train_bytes: values.iter().map(|v| v.len()).sum(),
        secs,
    })
}

#[derive(serde::Serialize)]
struct Row<'a> {
    dataset: &'a str,
    table: &'a str,
    column: &'a str,
    /// Training level of this codec's dictionary, or "self"/"none".
    train_level: &'a str,
    /// "full", "dedup" or "`subset<k>`".
    variant: &'a str,
    timed: bool,
    train_bytes: usize,
    train_secs: f64,
    dict_id: Option<String>,
    #[serde(flatten)]
    r: &'a ColumnResult,
}

/// What a result row was trained on.
#[derive(Clone, Copy)]
struct Origin<'a> {
    level: &'a str,
    variant: &'a str,
    trained: Option<&'a Trained>,
    is_presst: bool,
}

struct Out {
    json: Option<std::fs::File>,
    save_dicts: Option<PathBuf>,
}

impl Out {
    fn row(&mut self, t: &Target<'_>, o: Origin<'_>, r: &ColumnResult) -> Result<()> {
        let timed = r.scan_mbps > 0.0;
        if timed {
            println!(
                "  {:<26} {:<5} {:>7.3} {:>7.3} {:>9} {:>9.0} {:>9.0} {:>10.0}",
                r.codec,
                o.level,
                r.ratio,
                r.ratio_distinct,
                human(r.shared_bytes),
                r.compress_mbps,
                r.scan_mbps,
                r.lookup_ns
            );
        } else {
            println!(
                "  {:<26} {:<5} {:>7.3} {:>7.3} {:>9}   ({})",
                r.codec,
                o.level,
                r.ratio,
                r.ratio_distinct,
                human(r.shared_bytes),
                o.variant
            );
        }
        if let Some(f) = &mut self.json {
            let row = Row {
                dataset: t.dataset,
                table: t.table,
                column: t.column,
                train_level: o.level,
                variant: o.variant,
                timed,
                train_bytes: o.trained.map_or(0, |x| x.train_bytes),
                train_secs: o.trained.map_or(0.0, |x| x.secs),
                dict_id: o.trained.filter(|_| o.is_presst).map(|x| x.dict.id().to_string()),
                r,
            };
            writeln!(f, "{}", serde_json::to_string(&row)?)?;
        }
        Ok(())
    }

    fn save(&self, t: &Target<'_>, level: &str, dict: &Dictionary) -> Result<()> {
        if let Some(d) = &self.save_dicts {
            let name = format!("{}-{}-{}-{level}.pdict", t.dataset, t.table, t.column);
            std::fs::write(d.join(name), dict.to_bytes())?;
        }
        Ok(())
    }
}

/// Per-run state shared across columns.
struct Ctx<'a> {
    args: &'a Args,
    out: Out,
    pool: Vec<Source>,
    variance: HashSet<ColumnKey>,
    l4_cache: HashMap<String, Option<Trained>>,
    l5_cache: HashMap<String, Trained>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(d) = &args.save_dicts {
        std::fs::create_dir_all(d)?;
    }
    let tables = find_tables(&args.data)?;
    let mut ctx = Ctx {
        args: &args,
        out: Out {
            json: match &args.json {
                Some(p) => Some(
                    OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(p)
                        .context("opening json output")?,
                ),
                None => None,
            },
            save_dicts: args.save_dicts.clone(),
        },
        pool: if args.quick {
            Vec::new()
        } else {
            load_pool(&tables, &args.dbtext)?
        },
        variance: variance_columns(&tables, args.variance),
        l4_cache: HashMap::new(),
        l5_cache: HashMap::new(),
    };
    eprintln!("pool for L4/L5: {} source columns", ctx.pool.len());

    if !args.stress.is_empty() {
        let wanted = |m: &Meta| {
            (args.datasets.is_empty() || args.datasets.contains(&m.dataset))
                && (args.tables.is_empty() || args.tables.iter().any(|t| m.table.contains(t.as_str())))
        };
        let tables: Vec<(PathBuf, Meta)> = tables.into_iter().filter(|(_, m)| wanted(m)).collect();
        for mode in &args.stress {
            match mode.as_str() {
                "aging" => {
                    for (path, m) in &tables {
                        for c in &m.columns {
                            let t = Target {
                                dataset: &m.dataset,
                                table: &m.table,
                                column: &c.name,
                            };
                            stress::aging(&mut ctx, &t, &m.level, path.parent().expect("dir"))?;
                        }
                    }
                }
                "matrix" => {
                    let cols = stress::selected(&ctx, &tables);
                    stress::matrix(&mut ctx, &cols)?;
                }
                "sweep" => {
                    let cols = stress::selected(&ctx, &tables);
                    stress::sweep(&mut ctx, &cols)?;
                }
                "mix" => stress::mix(&mut ctx, &tables)?,
                other => anyhow::bail!("unknown stress test {other}"),
            }
        }
        return Ok(());
    }

    for (meta_path, meta) in &tables {
        if !args.datasets.is_empty() && !args.datasets.contains(&meta.dataset) {
            continue;
        }
        if !args.tables.is_empty() && !args.tables.iter().any(|t| meta.table.contains(t.as_str())) {
            continue;
        }
        // Leakage guard: training and test material must not overlap.
        assert!(
            meta.train_sources.iter().all(|s| !meta.test_sources.contains(s)),
            "{}: train and test sources overlap",
            meta.table
        );
        let dir = meta_path.parent().expect("table dir");
        for col in &meta.columns {
            let t = Target {
                dataset: &meta.dataset,
                table: &meta.table,
                column: &col.name,
            };
            run_column(&mut ctx, &t, &meta.level, dir)?;
        }
    }
    Ok(())
}

fn run_column(ctx: &mut Ctx<'_>, t: &Target<'_>, level: &str, dir: &Path) -> Result<()> {
    let args = ctx.args;
    let train_vals = read_vals(&dir.join(format!("{}.train.vals", t.column)))?;
    let mut test_vals = read_vals(&dir.join(format!("{}.test.vals", t.column)))?;
    if args.quick {
        test_vals.truncate(200_000);
    }
    let train_refs: Vec<&[u8]> = train_vals.iter().map(Vec::as_slice).collect();
    let test_refs: Vec<&[u8]> = test_vals.iter().map(Vec::as_slice).collect();
    println!(
        "\n{}/{}/{} [{level}]: train {:.1} MB, test {} values ({:.1} MB)",
        t.dataset,
        t.table,
        t.column,
        train_refs.iter().map(|v| v.len()).sum::<usize>() as f64 / 1e6,
        test_refs.len(),
        test_refs.iter().map(|v| v.len()).sum::<usize>() as f64 / 1e6
    );
    println!(
        "  {:<26} {:<5} {:>7} {:>7} {:>9} {:>9} {:>9} {:>10}",
        "codec", "train", "ratio", "strings", "shared", "comp MB/s", "scan MB/s", "lookup ns"
    );

    // Native level, fully timed. Every trained codec sees the same values.
    let ext = TrainOptions::extended();
    let native = train_all(&train_refs, &ext)?;
    ctx.out.save(t, level, &native.dict)?;
    let fsst_mode = train_all(&train_refs, &TrainOptions::fsst())?;
    timed_rows(ctx, t, level, &native, &fsst_mode, &test_refs)?;
    if args.quick {
        return Ok(());
    }

    // Duplicate control: drop test values that occur in the training data.
    let seen: HashSet<&[u8]> = train_refs.iter().copied().collect();
    let fresh: Vec<&[u8]> = test_refs.iter().copied().filter(|v| !seen.contains(v)).collect();
    if !fresh.is_empty() {
        let share = fresh.iter().map(|v| v.len()).sum::<usize>() as f64
            / test_refs.iter().map(|v| v.len()).sum::<usize>().max(1) as f64;
        println!("  -- unseen values only ({:.0}% of test bytes)", share * 100.0);
        ratio_rows(&mut ctx.out, t, level, "dedup", &native, &fresh)?;
        let o = Origin {
            level: "self",
            variant: "dedup",
            trained: None,
            is_presst: false,
        };
        ctx.out.row(t, o, &measure_ratio(&FsstRowGroup, &fresh))?;
    }

    // Other training levels: ratio only.
    let l0 = train_all(&spread(&test_refs, train_refs.len().max(1)), &ext)?;
    ctx.out.save(t, "L0", &l0.dict)?;
    ratio_rows(&mut ctx.out, t, "L0", "full", &l0, &test_refs)?;

    let key = format!("{}/{}/{:?}", t.dataset, t.table, kind_of(t.dataset, t.column));
    if !ctx.l4_cache.contains_key(&key) {
        let trained = match l4_sources(t, &ctx.pool) {
            Some((desc, sources)) => {
                eprintln!("  L4 training set: {desc} ({} columns)", sources.len());
                Some(train_all(&concat_sources(&sources), &ext)?)
            }
            None => None,
        };
        ctx.l4_cache.insert(key.clone(), trained);
    }
    if let Some(l4) = &ctx.l4_cache[&key] {
        ctx.out.save(t, "L4", &l4.dict)?;
        ratio_rows(&mut ctx.out, t, "L4", "full", l4, &test_refs)?;
    }

    if !ctx.l5_cache.contains_key(t.dataset) {
        let sources = l5_sources(t, &ctx.pool);
        eprintln!("  L5 training set: {} columns from other datasets", sources.len());
        let trained = train_all(&concat_sources(&sources), &ext)?;
        ctx.l5_cache.insert(t.dataset.to_owned(), trained);
    }
    let l5 = &ctx.l5_cache[t.dataset];
    ctx.out.save(t, "L5", &l5.dict)?;
    ratio_rows(&mut ctx.out, t, "L5", "full", l5, &test_refs)?;

    // Variance over training subsets and seeds, for the largest columns.
    if ctx
        .variance
        .contains(&(t.dataset.to_owned(), t.table.to_owned(), t.column.to_owned()))
    {
        for k in 0..3usize {
            let subset: Vec<Vec<u8>> = train_refs.iter().skip(k).step_by(3).map(|v| v.to_vec()).collect();
            let opts = TrainOptions {
                seed: 1000 + k as u64,
                ..TrainOptions::extended()
            };
            let dict = train(&Corpus::from_records(subset), &opts)?.dictionary;
            let r = measure_ratio(&PresstColumn::new("presst", &dict), &test_refs);
            let variant = format!("subset{k}");
            let o = Origin {
                level,
                variant: &variant,
                trained: None,
                is_presst: true,
            };
            ctx.out.row(t, o, &r)?;
        }
    }
    Ok(())
}

/// The fully timed codecs at the native training level.
fn timed_rows(
    ctx: &mut Ctx<'_>,
    t: &Target<'_>,
    level: &str,
    native: &Trained,
    fsst_mode: &Trained,
    test: &[&[u8]],
) -> Result<()> {
    let (runs, lookups) = if ctx.args.quick {
        (1, ctx.args.lookups)
    } else {
        (ctx.args.runs, ctx.args.lookups)
    };
    let o = |level, trained, is_presst| Origin {
        level,
        variant: "full",
        trained,
        is_presst,
    };
    let mut list: Vec<(Box<dyn ColumnCodec + '_>, Origin<'_>)> = vec![
        (
            Box::new(PresstColumn::new("presst", &native.dict)),
            o(level, Some(native), true),
        ),
        (
            Box::new(PresstColumn::new("presst (fsst-trained)", &fsst_mode.dict)),
            o(level, Some(fsst_mode), true),
        ),
        (Box::new(Shared(&native.fsst)), o(level, Some(native), false)),
        (Box::new(FsstRowGroup), o("self", None, false)),
        (Box::new(PerVector::new(Lz4)), o("none", None, false)),
        (Box::new(PerVector::new(Zstd::new(3))), o("none", None, false)),
    ];
    if let Some(z) = &native.zstd {
        list.push((Box::new(PerVector::new(ZstdRef(z))), o(level, Some(native), false)));
    }
    for (codec, origin) in &list {
        let r = measure_column(codec.as_ref(), test, runs, lookups);
        ctx.out.row(t, *origin, &r)?;
    }
    Ok(())
}

/// Ratio of the three trained codecs on `values`.
fn ratio_rows(out: &mut Out, t: &Target<'_>, level: &str, variant: &str, tr: &Trained, values: &[&[u8]]) -> Result<()> {
    let o = |is_presst| Origin {
        level,
        variant,
        trained: Some(tr),
        is_presst,
    };
    out.row(
        t,
        o(true),
        &measure_ratio(&PresstColumn::new("presst", &tr.dict), values),
    )?;
    out.row(t, o(false), &measure_ratio(&Shared(&tr.fsst), values))?;
    if let Some(z) = &tr.zstd {
        out.row(t, o(false), &measure_ratio(&PerVector::new(ZstdRef(z)), values))?;
    }
    Ok(())
}

/// Borrowing wrapper so a shared FSST table can be measured without moving it.
struct Shared<'a>(&'a FsstShared);

impl ColumnCodec for Shared<'_> {
    fn name(&self) -> String {
        self.0.name()
    }
    fn shared_bytes(&self) -> usize {
        self.0.shared_bytes()
    }
    fn encode(&self, values: &[&[u8]]) -> Encoded {
        self.0.encode(values)
    }
    fn scan_vector(&self, enc: &Encoded, vector: usize, out: &mut Vec<u8>) {
        self.0.scan_vector(enc, vector, out);
    }
    fn get(&self, enc: &Encoded, row: usize, out: &mut Vec<u8>) {
        self.0.get(enc, row, out);
    }
}

/// Borrowing wrapper for a trained zstd codec.
struct ZstdRef<'a>(&'a Zstd);

impl BlockCodec for ZstdRef<'_> {
    fn name(&self) -> String {
        self.0.name()
    }
    fn compress(&self, block: &[u8], out: &mut Vec<u8>) {
        self.0.compress(block, out);
    }
    fn decompress(&self, data: &[u8], len: usize, out: &mut Vec<u8>) {
        self.0.decompress(data, len, out);
    }
}

impl SharedBytes for ZstdRef<'_> {
    fn shared(&self) -> usize {
        self.0.dict_len()
    }
}

/// Mixes the sources with an equal byte share each, interleaved so that a
/// sample of the result covers all of them.
fn concat_sources<'a>(sources: &[&'a Source]) -> Vec<&'a [u8]> {
    let mut out = Vec::new();
    let longest = sources.iter().map(|s| s.sample.len()).max().unwrap_or(0);
    for i in 0..longest {
        for s in sources {
            if let Some(v) = s.sample.get(i) {
                out.push(v.as_slice());
            }
        }
    }
    out
}

/// Loads a `POOL_BYTES` sample of every column's training data, plus dbtext.
fn load_pool(tables: &[(PathBuf, Meta)], dbtext: &Path) -> Result<Vec<Source>> {
    let mut pool = Vec::new();
    for (path, meta) in tables {
        for col in &meta.columns {
            let vals = read_vals(&path.parent().expect("dir").join(format!("{}.train.vals", col.name)))?;
            pool.push(Source {
                dataset: meta.dataset.clone(),
                table: meta.table.clone(),
                kind: kind_of(&meta.dataset, &col.name),
                sample: sample_bytes(vals, POOL_BYTES),
            });
        }
    }
    if let Ok(dir) = std::fs::read_dir(dbtext) {
        for entry in dir {
            let p = entry?.path();
            let name = p.file_name().expect("name").to_string_lossy().into_owned();
            let data = std::fs::read(&p)?;
            let lines: Vec<Vec<u8>> = data
                .split(|&b| b == b'\n')
                .filter(|l| !l.is_empty())
                .map(<[u8]>::to_vec)
                .collect();
            pool.push(Source {
                dataset: "dbtext".into(),
                table: name.clone(),
                kind: kind_of("dbtext", &name),
                sample: sample_bytes(lines, POOL_BYTES),
            });
        }
    }
    Ok(pool)
}

/// At most about `max` bytes of values, evenly spread.
fn sample_bytes(vals: Vec<Vec<u8>>, max: usize) -> Vec<Vec<u8>> {
    let total: usize = vals.iter().map(Vec::len).sum();
    if total <= max {
        return vals;
    }
    let step = total.div_ceil(max);
    vals.into_iter().step_by(step).collect()
}

/// (dataset, table, column)
type ColumnKey = (String, String, String);

/// The `n` largest columns per dataset get the variance check.
fn variance_columns(tables: &[(PathBuf, Meta)], n: usize) -> HashSet<ColumnKey> {
    let mut by_dataset: HashMap<&str, Vec<(usize, ColumnKey)>> = HashMap::new();
    for (_, m) in tables {
        for c in &m.columns {
            by_dataset
                .entry(&m.dataset)
                .or_default()
                .push((c.test_bytes, (m.dataset.clone(), m.table.clone(), c.name.clone())));
        }
    }
    let mut out = HashSet::new();
    for (_, mut cols) in by_dataset {
        cols.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        out.extend(cols.into_iter().take(n).map(|(_, k)| k));
    }
    out
}

/// At most `n` values, evenly spread over `values`.
fn spread<'a>(values: &[&'a [u8]], n: usize) -> Vec<&'a [u8]> {
    if values.len() <= n {
        return values.to_vec();
    }
    (0..n).map(|i| values[i * values.len() / n]).collect()
}

fn find_tables(root: &Path) -> Result<Vec<(PathBuf, Meta)>> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))? {
            let p = entry?.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "meta.json") {
                let meta: Meta = serde_json::from_slice(&std::fs::read(&p)?)?;
                out.push((p, meta));
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

fn human(bytes: usize) -> String {
    match bytes {
        0 => "-".into(),
        b if b >= 1024 => format!("{:.1} KiB", b as f64 / 1024.0),
        b => format!("{b} B"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn src(dataset: &str, table: &str, column: &str) -> (Source, String) {
        let s = Source {
            dataset: dataset.into(),
            table: table.into(),
            kind: kind_of(dataset, column),
            sample: vec![b"x".to_vec()],
        };
        (s, column.into())
    }

    /// No L4 or L5 training set may contain the column under test.
    #[test]
    fn no_leakage_into_l4_l5() {
        let entries = vec![
            src("loghub", "HDFS", "line"),
            src("loghub", "BGL", "line"),
            src("clickbench", "hits", "URL"),
            src("clickbench", "hits", "Title"),
            src("publicbi", "Taxpayer", "hcpcs_description"),
            src("publicbi", "Medicare1", "website_url"),
            src("dbtext", "urls", "urls"),
            src("dbtext", "wiki", "wiki"),
        ];
        let columns: Vec<String> = entries.iter().map(|(_, c)| c.clone()).collect();
        let pool: Vec<Source> = entries.into_iter().map(|(s, _)| s).collect();
        for (s, column) in pool.iter().zip(&columns) {
            let t = Target {
                dataset: &s.dataset,
                table: &s.table,
                column,
            };
            if let Some((_, l4)) = l4_sources(&t, &pool) {
                assert!(
                    l4.iter().all(|x| x.table != s.table || x.dataset != s.dataset),
                    "L4 leaks {}",
                    s.table
                );
            }
            assert!(
                l5_sources(&t, &pool).iter().all(|x| x.dataset != s.dataset),
                "L5 leaks {}",
                s.dataset
            );
        }
        let url = Target {
            dataset: "clickbench",
            table: "hits",
            column: "URL",
        };
        let (_, l4) = l4_sources(&url, &pool).unwrap();
        assert!(l4.iter().all(|s| s.kind == Kind::Url && s.dataset != "clickbench"));
        assert_eq!(l4.len(), 2, "Public BI URL column and dbtext urls");
    }
}
