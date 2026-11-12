//! `cargo xtask e2ereport <results dir> [-o report.md]`: Markdown summary of the end-to-end
//! benchmarks written by `e2e/run-vortex.sh` and `e2e/run-duckdb.sh`.
//!
//! Layout read:
//! * `vortex/<suite>/ingest.tsv` (format, seconds, bytes) and `vortex/<suite>/<engine>.json`
//!   (gh-json lines from Vortex's benchmark binaries: `name` = `<dataset>_qNN/<engine>:<format>`,
//!   `value` = median nanoseconds).
//! * `duckdb/<suite>/<variant>.tsv` (benchmark_runner output: name, run, seconds) and
//!   `duckdb/<suite>/<variant>.load.txt`.
//!
//! Speedups are per query (baseline time / presst time); a suite's figure is their geometric mean,
//! so every query counts equally regardless of its length.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

/// query -> format -> median seconds
type Times = BTreeMap<String, BTreeMap<String, f64>>;

pub fn run(args: &[String]) -> Result<()> {
    let mut dir = None;
    let mut out = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-o" => out = Some(PathBuf::from(it.next().context("-o needs a path")?)),
            s if s.starts_with('-') => bail!("unknown option {s}"),
            s => dir = Some(PathBuf::from(s)),
        }
    }
    let dir = dir.context("usage: cargo xtask e2ereport <results dir> [-o report.md]")?;
    let mut md = String::new();
    vortex_section(&dir.join("vortex"), &mut md)?;
    duckdb_section(&dir.join("duckdb"), &mut md)?;
    match out {
        Some(p) => std::fs::write(p, md)?,
        None => print!("{md}"),
    }
    Ok(())
}

fn subdirs(dir: &Path) -> Result<Vec<PathBuf>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    v.sort();
    Ok(v)
}

fn name(p: &Path) -> String {
    p.file_name().expect("dir name").to_string_lossy().into_owned()
}

fn geomean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    (xs.iter().map(|x| x.ln()).sum::<f64>() / xs.len() as f64).exp()
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(f64::total_cmp);
    match xs.len() {
        0 => f64::NAN,
        n if n % 2 == 1 => xs[n / 2],
        n => f64::midpoint(xs[n / 2 - 1], xs[n / 2]),
    }
}

/// Per-query comparison of `base` and `new`: (queries, total base s, total new s, geomean speedup,
/// queries more than 5% faster, more than 5% slower, slowest-ratio query).
struct Compare {
    queries: usize,
    base_total: f64,
    new_total: f64,
    speedup: f64,
    faster: usize,
    slower: usize,
    worst: Option<(String, f64)>,
    best: Option<(String, f64)>,
}

fn compare(times: &Times, base: &str, new: &str) -> Compare {
    let mut ratios = Vec::new();
    let mut c = Compare {
        queries: 0,
        base_total: 0.0,
        new_total: 0.0,
        speedup: f64::NAN,
        faster: 0,
        slower: 0,
        worst: None,
        best: None,
    };
    for (q, by_format) in times {
        let (Some(&b), Some(&n)) = (by_format.get(base), by_format.get(new)) else {
            continue;
        };
        if b <= 0.0 || n <= 0.0 {
            continue;
        }
        let r = b / n;
        ratios.push(r);
        c.queries += 1;
        c.base_total += b;
        c.new_total += n;
        if r > 1.05 {
            c.faster += 1;
        } else if r < 1.0 / 1.05 {
            c.slower += 1;
        }
        if c.worst.as_ref().is_none_or(|(_, w)| r < *w) {
            c.worst = Some((q.clone(), r));
        }
        if c.best.as_ref().is_none_or(|(_, w)| r > *w) {
            c.best = Some((q.clone(), r));
        }
    }
    c.speedup = geomean(&ratios);
    c
}

fn fmt_extreme(e: Option<&(String, f64)>) -> String {
    e.map_or_else(|| "–".into(), |(q, r)| format!("{q} {r:.2}×"))
}

/// Reads gh-json lines: name `<dataset>_qNN/<engine>:<format>`, value in ns.
fn read_gh_json(path: &Path) -> Result<Times> {
    let mut times = Times::new();
    if !path.exists() {
        return Ok(times);
    }
    for line in std::fs::read_to_string(path)?.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value = serde_json::from_str(line).with_context(|| format!("{}: {line}", path.display()))?;
        if v.get("unit").and_then(|u| u.as_str()) != Some("ns") {
            continue;
        }
        let (Some(name), Some(ns)) = (
            v.get("name").and_then(|n| n.as_str()),
            v.get("value").and_then(serde_json::Value::as_f64),
        ) else {
            continue;
        };
        let Some((query, target)) = name.split_once('/') else {
            continue;
        };
        let format = target.split_once(':').map_or(target, |(_, f)| f);
        times
            .entry(query.to_owned())
            .or_default()
            .insert(format.to_owned(), ns / 1e9);
    }
    Ok(times)
}

/// format -> (write seconds, bytes)
type Ingest = BTreeMap<String, (Option<f64>, Option<u64>)>;

fn read_ingest(path: &Path) -> Result<Ingest> {
    let mut m = BTreeMap::new();
    if !path.exists() {
        return Ok(m);
    }
    for line in std::fs::read_to_string(path)?.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() >= 3 {
            m.insert(f[0].to_owned(), (f[1].parse().ok(), f[2].trim().parse().ok()));
        }
    }
    Ok(m)
}

fn mb(b: Option<u64>) -> String {
    b.map_or_else(|| "–".into(), |b| format!("{:.1}", b as f64 / 1e6))
}

fn vortex_section(dir: &Path, md: &mut String) -> Result<()> {
    let suites = subdirs(dir)?;
    if suites.is_empty() {
        return Ok(());
    }
    md.push_str("## Vortex: default files vs. vortex-presst\n\n");
    md.push_str(
        "Speedup = default Vortex time / vortex-presst time per query, geometric mean over the suite's queries.\n\n",
    );
    md.push_str("| Suite | Engine | Queries | Vortex total (s) | presst total (s) | Speedup | >5% faster | >5% slower | Worst query | Best query |\n");
    md.push_str("|---|---|---:|---:|---:|---:|---:|---:|---|---|\n");
    for s in &suites {
        for engine in ["datafusion", "duckdb"] {
            let times = read_gh_json(&s.join(format!("{engine}.json")))?;
            let c = compare(&times, "vortex-file-compressed", "vortex-presst");
            if c.queries == 0 {
                continue;
            }
            let _ = writeln!(
                md,
                "| {} | {engine} | {} | {:.2} | {:.2} | {:.3}× | {} | {} | {} | {} |",
                name(s),
                c.queries,
                c.base_total,
                c.new_total,
                c.speedup,
                c.faster,
                c.slower,
                fmt_extreme(c.worst.as_ref()),
                fmt_extreme(c.best.as_ref())
            );
        }
    }
    md.push_str("\n### Size and write time\n\n");
    md.push_str("| Suite | Parquet (MB) | Vortex (MB) | vortex-presst (MB) | Size change | Vortex write (s) | presst write (s) |\n");
    md.push_str("|---|---:|---:|---:|---:|---:|---:|\n");
    for s in &suites {
        let ing = read_ingest(&s.join("ingest.tsv"))?;
        let get = |f: &str| ing.get(f).copied().unwrap_or((None, None));
        let (pq, vx, pr) = (get("parquet"), get("vortex"), get("vortex-presst"));
        let change = match (vx.1, pr.1) {
            (Some(a), Some(b)) if a > 0 => format!("{:+.2}%", (b as f64 / a as f64 - 1.0) * 100.0),
            _ => "–".into(),
        };
        let secs = |x: Option<f64>| x.map_or_else(|| "–".into(), |x| format!("{x:.1}"));
        let _ = writeln!(
            md,
            "| {} | {} | {} | {} | {change} | {} | {} |",
            name(s),
            mb(pq.1),
            mb(vx.1),
            mb(pr.1),
            secs(vx.0),
            secs(pr.0)
        );
    }
    md.push('\n');
    Ok(())
}

/// benchmark_runner output: `name<TAB>run<TAB>seconds` lines; median per benchmark.
fn read_runner(path: &Path) -> Result<BTreeMap<String, f64>> {
    let mut runs: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    for line in std::fs::read_to_string(path)?.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() >= 3
            && let Ok(t) = f[2].trim().parse::<f64>()
        {
            runs.entry(f[0].to_owned()).or_default().push(t);
        }
    }
    Ok(runs.into_iter().map(|(k, v)| (k, median(v))).collect())
}

fn load_info(path: &Path) -> (Option<f64>, Option<u64>) {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let field = |prefix: &str| {
        text.lines()
            .find_map(|l| l.strip_prefix(prefix))
            .map(|v| v.trim().to_owned())
    };
    (
        field("first pass (includes loading) seconds:").and_then(|v| v.parse().ok()),
        field("database bytes:").and_then(|v| v.parse().ok()),
    )
}

fn duckdb_section(dir: &Path, md: &mut String) -> Result<()> {
    let suites = subdirs(dir)?;
    if suites.is_empty() {
        return Ok(());
    }
    md.push_str("## DuckDB native storage: DICT_FSST vs. DICT_FSST with presst tables\n\n");
    md.push_str("| Suite | Queries | DICT_FSST total (s) | presst total (s) | Speedup | >5% faster | >5% slower | Worst query | Best query | DB size change | First run incl. load (s): FSST / presst |\n");
    md.push_str("|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|\n");
    for s in &suites {
        let mut times = Times::new();
        for variant in ["dict_fsst", "presst"] {
            for (q, t) in read_runner(&s.join(format!("{variant}.tsv")))? {
                times.entry(q).or_default().insert(variant.to_owned(), t);
            }
        }
        let c = compare(&times, "dict_fsst", "presst");
        let (fl, fb) = load_info(&s.join("dict_fsst.load.txt"));
        let (pl, pb) = load_info(&s.join("presst.load.txt"));
        let change = match (fb, pb) {
            (Some(a), Some(b)) if a > 0 => format!("{:+.2}%", (b as f64 / a as f64 - 1.0) * 100.0),
            _ => "–".into(),
        };
        let secs = |x: Option<f64>| x.map_or_else(|| "–".into(), |x| format!("{x:.1}"));
        let _ = writeln!(
            md,
            "| {} | {} | {:.2} | {:.2} | {:.3}× | {} | {} | {} | {} | {change} | {} / {} |",
            name(s),
            c.queries,
            c.base_total,
            c.new_total,
            c.speedup,
            c.faster,
            c.slower,
            fmt_extreme(c.worst.as_ref()),
            fmt_extreme(c.best.as_ref()),
            secs(fl),
            secs(pl)
        );
    }
    md.push('\n');
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_and_geomean() {
        let mut t = Times::new();
        t.entry("q1".into())
            .or_default()
            .extend([("a".into(), 2.0), ("b".into(), 1.0)]);
        t.entry("q2".into())
            .or_default()
            .extend([("a".into(), 1.0), ("b".into(), 2.0)]);
        t.entry("q3".into()).or_default().insert("a".into(), 1.0);
        let c = compare(&t, "a", "b");
        assert_eq!(c.queries, 2);
        assert!((c.speedup - 1.0).abs() < 1e-12);
        assert_eq!((c.faster, c.slower), (1, 1));
        assert_eq!(c.best.unwrap().0, "q1");
    }

    #[test]
    fn parses_gh_json_and_runner_output() {
        let dir = std::env::temp_dir().join("presst-e2ereport-test");
        std::fs::create_dir_all(&dir).unwrap();
        let gh = dir.join("df.json");
        std::fs::write(
            &gh,
            r#"{"name":"tpch_q01/datafusion:vortex-file-compressed","unit":"ns","value":2000000000}
{"name":"tpch_q01/datafusion:vortex-presst","unit":"ns","value":1000000000}
{"name":"memory","unit":"bytes","value":5}
"#,
        )
        .unwrap();
        let t = read_gh_json(&gh).unwrap();
        assert_eq!(t["tpch_q01"]["vortex-presst"], 1.0);
        let tsv = dir.join("r.tsv");
        std::fs::write(
            &tsv,
            "name\trun\ttiming\nQ01\t1\t0.5\nQ01\t2\t0.7\nQ01\t3\t0.6\nQ02\t1\tERROR\n",
        )
        .unwrap();
        let r = read_runner(&tsv).unwrap();
        assert_eq!(r.len(), 1);
        assert!((r["Q01"] - 0.6).abs() < 1e-12);
    }
}
