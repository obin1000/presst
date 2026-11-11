//! `cargo xtask colreport <results.jsonl>... [--svg <out.svg>]`: turns
//! `presst-colbench` results into Markdown tables (and an SVG chart).
//!
//! Aggregates are weighted by bytes: ratios are total input over total output,
//! throughput is total bytes over total time. Lookup latency is the geometric
//! mean over columns.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use anyhow::{Context, Result};

#[derive(serde::Deserialize, Clone)]
struct Row {
    dataset: String,
    table: String,
    column: String,
    train_level: String,
    variant: String,
    timed: bool,
    codec: String,
    input_bytes: u64,
    distinct_bytes: u64,
    index_bytes: u64,
    compressed_bytes: u64,
    compress_mbps: f64,
    scan_mbps: f64,
    lookup_ns: f64,
    shared_bytes: usize,
}

/// Running totals for one group of rows.
#[derive(Default, Clone)]
struct Agg {
    input: u64,
    distinct: u64,
    index: u64,
    compressed: u64,
    compress_secs: f64,
    scan_secs: f64,
    log_lookup: f64,
    columns: usize,
    max_shared: usize,
}

impl Agg {
    fn add(&mut self, r: &Row) {
        let mb = r.distinct_bytes as f64 / 1e6;
        self.input += r.input_bytes;
        self.distinct += r.distinct_bytes;
        self.index += r.index_bytes;
        self.compressed += r.compressed_bytes;
        if r.timed {
            self.compress_secs += mb / r.compress_mbps.max(1e-9);
            self.scan_secs += mb / r.scan_mbps.max(1e-9);
            self.log_lookup += r.lookup_ns.max(1.0).ln();
        }
        self.columns += 1;
        self.max_shared = self.max_shared.max(r.shared_bytes);
    }
    fn ratio(&self) -> f64 {
        self.input as f64 / (self.compressed + self.index).max(1) as f64
    }
    fn strings(&self) -> f64 {
        self.distinct as f64 / self.compressed.max(1) as f64
    }
    fn compress(&self) -> f64 {
        self.distinct as f64 / 1e6 / self.compress_secs.max(1e-12)
    }
    fn scan(&self) -> f64 {
        self.distinct as f64 / 1e6 / self.scan_secs.max(1e-12)
    }
    fn lookup(&self) -> f64 {
        (self.log_lookup / self.columns.max(1) as f64).exp()
    }
}

fn codec_order(c: &str) -> usize {
    [
        "presst",
        "presst (fsst-trained)",
        "fsst (shared table)",
        "fsst (table per row group)",
        "lz4 /vector",
        "zstd -3 /vector",
        "zstd -3 +dict110K /vector",
    ]
    .iter()
    .position(|x| *x == c)
    .unwrap_or(99)
}

fn sorted_codecs<'a>(it: impl Iterator<Item = &'a String>) -> Vec<String> {
    let mut v: Vec<String> = it.cloned().collect::<BTreeSet<_>>().into_iter().collect();
    v.sort_by_key(|c| (codec_order(c), c.clone()));
    v
}

fn ns(x: f64) -> String {
    if x >= 1e6 {
        format!("{:.1} ms", x / 1e6)
    } else if x >= 1e3 {
        format!("{:.1} µs", x / 1e3)
    } else {
        format!("{x:.0} ns")
    }
}

pub fn run(args: &[String]) -> Result<()> {
    let mut rows = Vec::new();
    let mut svg_out = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--svg" {
            svg_out = Some(it.next().context("--svg needs a path")?.clone());
            continue;
        }
        let text = std::fs::read_to_string(a).with_context(|| format!("reading {a}"))?;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            rows.push(serde_json::from_str::<Row>(line)?);
        }
    }
    let mut md = String::new();
    let datasets: BTreeSet<String> = rows.iter().map(|r| r.dataset.clone()).collect();
    let native = |r: &Row| r.variant == "full" && r.timed;

    // 1. Headline: native level, timed.
    writeln!(md, "## Results at the native training level\n")?;
    writeln!(
        md,
        "Column ratio = all values / (compressed distinct values + index). String ratio = distinct values / compressed. \
         Speeds are over the distinct values; lookups fetch one random row.\n"
    )?;
    let mut chart: Vec<(String, String, f64, f64)> = Vec::new();
    for ds in datasets.iter().map(String::as_str).chain(["all"]) {
        let sel: Vec<&Row> = rows
            .iter()
            .filter(|r| native(r) && (ds == "all" || r.dataset == ds))
            .collect();
        if sel.is_empty() {
            continue;
        }
        let ncols = sel
            .iter()
            .map(|r| (&r.dataset, &r.table, &r.column))
            .collect::<BTreeSet<_>>()
            .len();
        writeln!(md, "**{ds}** ({ncols} columns)\n")?;
        writeln!(
            md,
            "| Codec | Column ratio | String ratio | Compress MB/s | Scan MB/s | Lookup | Shared dict |"
        )?;
        writeln!(md, "|---|---:|---:|---:|---:|---:|---:|")?;
        for c in sorted_codecs(sel.iter().map(|r| &r.codec)) {
            let mut a = Agg::default();
            sel.iter().filter(|r| r.codec == c).for_each(|r| a.add(r));
            let shared = if a.max_shared == 0 {
                "–".into()
            } else {
                format!("≤ {:.1} KiB", a.max_shared as f64 / 1024.0)
            };
            writeln!(
                md,
                "| {c} | {:.2} | {:.2} | {:.0} | {:.0} | {} | {shared} |",
                a.ratio(),
                a.strings(),
                a.compress(),
                a.scan(),
                ns(a.lookup())
            )?;
            if ds != "all" {
                chart.push((ds.to_owned(), c.clone(), a.scan(), a.strings()));
            }
        }
        writeln!(md)?;
    }

    // 2. Training levels.
    let level_rows: Vec<&Row> = rows
        .iter()
        .filter(|r| r.variant == "full" && r.train_level != "self" && r.train_level != "none")
        .collect();
    if level_rows.iter().any(|r| r.train_level == "L0") {
        writeln!(md, "## Training distance\n")?;
        writeln!(
            md,
            "String ratio of the trained codecs by how far their training data is from the test data. Gap = ratio / ratio at L0 (trained on the test data itself).\n"
        )?;
        for ds in &datasets {
            writeln!(md, "**{ds}**\n")?;
            writeln!(
                md,
                "| Codec | L0 (oracle) | native | L4 (other tables) | L5 (other datasets) |"
            )?;
            writeln!(md, "|---|---:|---:|---:|---:|")?;
            let sel: Vec<&&Row> = level_rows
                .iter()
                .filter(|r| &r.dataset == ds && r.codec != "presst (fsst-trained)")
                .collect();
            // Only columns that have all of L0, native and L5, so levels compare like with like.
            let cols_with = |lvl: &dyn Fn(&str) -> bool| -> BTreeSet<(String, String)> {
                sel.iter()
                    .filter(|r| lvl(&r.train_level))
                    .map(|r| (r.table.clone(), r.column.clone()))
                    .collect()
            };
            let l0 = cols_with(&|l| l == "L0");
            let nat = cols_with(&|l| l.starts_with('L') && !matches!(l, "L0" | "L4" | "L5"));
            let l5 = cols_with(&|l| l == "L5");
            let common: BTreeSet<_> = l0
                .intersection(&nat)
                .cloned()
                .collect::<BTreeSet<_>>()
                .intersection(&l5)
                .cloned()
                .collect();
            for c in sorted_codecs(sel.iter().map(|r| &r.codec)) {
                let agg = |lvl: &dyn Fn(&str) -> bool, need_common: bool| {
                    let mut a = Agg::default();
                    sel.iter()
                        .filter(|r| r.codec == c && lvl(&r.train_level))
                        .filter(|r| !need_common || common.contains(&(r.table.clone(), r.column.clone())))
                        .for_each(|r| a.add(r));
                    a
                };
                let a0 = agg(&|l| l == "L0", true);
                let an = agg(&|l| l.starts_with('L') && !matches!(l, "L0" | "L4" | "L5"), true);
                let a4 = agg(&|l| l == "L4", false);
                let a5 = agg(&|l| l == "L5", true);
                let cell = |a: &Agg| {
                    if a.columns == 0 {
                        "–".to_owned()
                    } else {
                        format!(
                            "{:.2} ({:.0}%)",
                            a.strings(),
                            100.0 * a.strings() / a0.strings().max(1e-9)
                        )
                    }
                };
                writeln!(
                    md,
                    "| {c} | {:.2} | {} | {} | {} |",
                    a0.strings(),
                    cell(&an),
                    cell(&a4),
                    cell(&a5)
                )?;
            }
            writeln!(md)?;
        }
    }

    // 3. Duplicate control.
    if rows.iter().any(|r| r.variant == "dedup") {
        writeln!(md, "## Duplicate control\n")?;
        writeln!(
            md,
            "String ratio on all test values vs. only test values that never occur in the training data.\n"
        )?;
        writeln!(md, "| Dataset | Codec | All test values | Unseen values only |")?;
        writeln!(md, "|---|---|---:|---:|")?;
        for ds in &datasets {
            let dedup: Vec<&Row> = rows
                .iter()
                .filter(|r| &r.dataset == ds && r.variant == "dedup")
                .collect();
            let cols: BTreeSet<(String, String)> = dedup.iter().map(|r| (r.table.clone(), r.column.clone())).collect();
            for c in sorted_codecs(dedup.iter().map(|r| &r.codec)) {
                let mut all = Agg::default();
                let mut fresh = Agg::default();
                rows.iter()
                    .filter(|r| {
                        &r.dataset == ds
                            && r.codec == c
                            && native(r)
                            && cols.contains(&(r.table.clone(), r.column.clone()))
                    })
                    .for_each(|r| all.add(r));
                dedup.iter().filter(|r| r.codec == c).for_each(|r| fresh.add(r));
                writeln!(md, "| {ds} | {c} | {:.2} | {:.2} |", all.strings(), fresh.strings())?;
            }
        }
        writeln!(md)?;
    }

    // 4. Variance over training subsets and seeds.
    let subsets: Vec<&Row> = rows.iter().filter(|r| r.variant.starts_with("subset")).collect();
    if !subsets.is_empty() {
        writeln!(md, "## Variance over training subsets and seeds (presst)\n")?;
        writeln!(
            md,
            "| Column | Full training set | 3 disjoint thirds, 3 seeds (min – max) |"
        )?;
        writeln!(md, "|---|---:|---:|")?;
        let cols: BTreeSet<(String, String, String)> = subsets
            .iter()
            .map(|r| (r.dataset.clone(), r.table.clone(), r.column.clone()))
            .collect();
        for (d, t, c) in cols {
            let full = rows
                .iter()
                .find(|r| r.dataset == d && r.table == t && r.column == c && r.codec == "presst" && native(r));
            let vals: Vec<f64> = subsets
                .iter()
                .filter(|r| r.dataset == d && r.table == t && r.column == c)
                .map(|r| r.distinct_bytes as f64 / r.compressed_bytes.max(1) as f64)
                .collect();
            let (lo, hi) = vals
                .iter()
                .fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
            let f = full.map_or(0.0, |r| r.distinct_bytes as f64 / r.compressed_bytes.max(1) as f64);
            writeln!(md, "| {d}/{t}/{c} | {f:.3} | {lo:.3} – {hi:.3} |")?;
        }
        writeln!(md)?;
    }

    // 5. Stress tests.
    stress_tables(&mut md, &rows)?;

    print!("{md}");
    if let Some(path) = svg_out {
        std::fs::write(&path, svg(&chart))?;
        eprintln!("wrote {path}");
    }
    Ok(())
}

fn stress_tables(md: &mut String, rows: &[Row]) -> Result<()> {
    // Aging.
    let aging: Vec<&Row> = rows.iter().filter(|r| r.variant.starts_with("age:")).collect();
    if !aging.is_empty() {
        let buckets = ["<1h", "1h-1d", "1d-7d", "7d-30d", "30d-180d", ">180d"];
        writeln!(md, "## Aging: string ratio by time since the training period\n")?;
        writeln!(
            md,
            "Dictionaries trained once on the past; FSST per row group retrains on every row group and is the non-aging reference.\n"
        )?;
        writeln!(md, "| Codec | {} |", buckets.join(" | "))?;
        writeln!(
            md,
            "|---|{}|",
            buckets.iter().map(|_| "---:").collect::<Vec<_>>().join("|")
        )?;
        for c in sorted_codecs(aging.iter().map(|r| &r.codec)) {
            let cells: Vec<String> = buckets
                .iter()
                .map(|b| {
                    let mut a = Agg::default();
                    aging
                        .iter()
                        .filter(|r| r.codec == c && r.variant == format!("age:{b}"))
                        .for_each(|r| a.add(r));
                    if a.columns == 0 {
                        "–".into()
                    } else {
                        format!("{:.2} ({})", a.strings(), a.columns)
                    }
                })
                .collect();
            writeln!(md, "| {c} | {} |", cells.join(" | "))?;
        }
        writeln!(md, "\n(n) = number of columns with data in that window.\n")?;
    }
    // Size sweep.
    let sweep: Vec<&Row> = rows.iter().filter(|r| r.variant.starts_with("size:")).collect();
    if !sweep.is_empty() {
        // Only columns that have every training size, so each size averages the same columns.
        let all_sizes: BTreeSet<u64> = sweep.iter().filter_map(|r| r.variant[5..].parse().ok()).collect();
        let mut by_col: BTreeMap<(String, String, String), BTreeSet<u64>> = BTreeMap::new();
        for r in &sweep {
            if let Ok(sz) = r.variant[5..].parse() {
                by_col
                    .entry((r.dataset.clone(), r.table.clone(), r.column.clone()))
                    .or_default()
                    .insert(sz);
            }
        }
        // Drop the largest sizes until at least half of the columns have all of them.
        let mut sizes = all_sizes.clone();
        while sizes.len() > 1 && by_col.values().filter(|s| sizes.is_subset(s)).count() * 2 < by_col.len() {
            let last = *sizes.iter().next_back().expect("non-empty");
            sizes.remove(&last);
        }
        let keep: BTreeSet<(String, String, String)> = by_col
            .iter()
            .filter(|(_, s)| sizes.is_subset(s))
            .map(|(k, _)| k.clone())
            .collect();
        let sweep: Vec<&Row> = sweep
            .into_iter()
            .filter(|r| keep.contains(&(r.dataset.clone(), r.table.clone(), r.column.clone())))
            .collect();
        writeln!(md, "## Training-size sweep: string ratio by bytes of training data\n")?;
        writeln!(
            md,
            "Prefix of each column's training data (in time or row order); the same {} columns at every size.\n",
            keep.len()
        )?;
        writeln!(
            md,
            "| Codec | {} |",
            sizes.iter().map(|s| human(*s)).collect::<Vec<_>>().join(" | ")
        )?;
        writeln!(
            md,
            "|---|{}|",
            sizes.iter().map(|_| "---:").collect::<Vec<_>>().join("|")
        )?;
        for c in sorted_codecs(sweep.iter().map(|r| &r.codec)) {
            let cells: Vec<String> = sizes
                .iter()
                .map(|s| {
                    let mut a = Agg::default();
                    sweep
                        .iter()
                        .filter(|r| r.codec == c && r.variant == format!("size:{s}"))
                        .for_each(|r| a.add(r));
                    if a.columns == 0 {
                        "–".into()
                    } else {
                        format!("{:.2}", a.strings())
                    }
                })
                .collect();
            writeln!(md, "| {c} | {} |", cells.join(" | "))?;
        }
        writeln!(md)?;
    }
    // Matrix.
    let matrix: Vec<&Row> = rows.iter().filter(|r| r.variant.starts_with("from:")).collect();
    if !matrix.is_empty() {
        let targets: Vec<String> = matrix
            .iter()
            .map(|r| format!("{}/{}/{}", r.dataset, r.table, r.column))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        writeln!(
            md,
            "## Cross-column matrix: presst string ratio, dictionary from row applied to column\n"
        )?;
        writeln!(
            md,
            "| Dictionary from \\ applied to | {} |",
            targets.iter().map(|t| short(t)).collect::<Vec<_>>().join(" | ")
        )?;
        writeln!(
            md,
            "|---|{}|",
            targets.iter().map(|_| "---:").collect::<Vec<_>>().join("|")
        )?;
        for from in &targets {
            let cells: Vec<String> = targets
                .iter()
                .map(|to| {
                    matrix
                        .iter()
                        .find(|r| {
                            format!("{}/{}/{}", r.dataset, r.table, r.column) == *to
                                && r.variant == format!("from:{from}")
                        })
                        .map_or("–".into(), |r| {
                            format!("{:.2}", r.distinct_bytes as f64 / r.compressed_bytes.max(1) as f64)
                        })
                })
                .collect();
            writeln!(md, "| {} | {} |", short(from), cells.join(" | "))?;
        }
        writeln!(md)?;
    }
    // Mix.
    let mix: Vec<&Row> = rows.iter().filter(|r| r.variant.starts_with("mix:")).collect();
    if !mix.is_empty() {
        writeln!(
            md,
            "## Generic dictionary: sensitivity to the training mix (presst string ratio)\n"
        )?;
        writeln!(md, "| Target dataset | Training mix | String ratio |")?;
        writeln!(md, "|---|---|---:|")?;
        let mut groups: BTreeMap<(String, String), Agg> = BTreeMap::new();
        for r in &mix {
            groups
                .entry((r.dataset.clone(), r.variant[4..].to_owned()))
                .or_default()
                .add(r);
        }
        for ((ds, m), a) in groups {
            writeln!(md, "| {ds} | {m} | {:.2} |", a.strings())?;
        }
        writeln!(md)?;
    }
    Ok(())
}

fn short(s: &str) -> String {
    let parts: Vec<&str> = s.split('/').collect();
    match parts.as_slice() {
        ["loghub", t, _] => (*t).to_owned(),
        [_, "hits", c] => (*c).to_owned(),
        [_, t, c] => format!("{t}.{c}"),
        _ => s.to_owned(),
    }
}

fn human(b: u64) -> String {
    if b >= 1 << 20 {
        format!("{} MiB", b >> 20)
    } else {
        format!("{} KiB", b >> 10)
    }
}

/// Scatter plot: scan speed (x, log scale) vs. string ratio (y), one marker
/// per codec and dataset. Plain SVG, readable in light and dark mode.
fn svg(points: &[(String, String, f64, f64)]) -> String {
    let (w, h, pad) = (760.0, 440.0, 60.0);
    let xmin = points
        .iter()
        .map(|p| p.2)
        .fold(f64::MAX, f64::min)
        .max(10.0)
        .log10()
        .floor();
    let xmax = points
        .iter()
        .map(|p| p.2)
        .fold(0.0, f64::max)
        .log10()
        .ceil()
        .max(xmin + 1.0);
    let ymax = points.iter().map(|p| p.3).fold(0.0, f64::max).ceil().max(1.0);
    let x = |v: f64| pad + (v.max(1.0).log10() - xmin) / (xmax - xmin) * (w - 2.0 * pad);
    let y = |v: f64| h - pad - v / ymax * (h - 2.0 * pad);
    let colors = [
        "#2a6fdb", "#7aa6f0", "#e07b39", "#c0392b", "#7f8c8d", "#27ae60", "#8e44ad",
    ];
    let codecs = sorted_codecs(points.iter().map(|p| &p.1));
    let datasets: Vec<String> = points
        .iter()
        .map(|p| p.0.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let shapes = ["circle", "square", "triangle"];
    let mut s = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif" font-size="12"><rect width="{w}" height="{h}" fill="white"/>"#
    );
    for e in (xmin as i32)..=(xmax as i32) {
        let xv = x(10f64.powi(e));
        let _ = write!(
            s,
            r##"<line x1="{xv}" y1="{pad}" x2="{xv}" y2="{}" stroke="#ddd"/><text x="{xv}" y="{}" text-anchor="middle" fill="#333">{}</text>"##,
            h - pad,
            h - pad + 18.0,
            10f64.powi(e)
        );
    }
    for k in 0..=(ymax as i32) {
        let yv = y(f64::from(k));
        let _ = write!(
            s,
            r##"<line x1="{pad}" y1="{yv}" x2="{}" y2="{yv}" stroke="#eee"/><text x="{}" y="{}" text-anchor="end" fill="#333">{k}</text>"##,
            w - pad,
            pad - 8.0,
            yv + 4.0
        );
    }
    let _ = write!(
        s,
        r##"<text x="{}" y="{}" text-anchor="middle" fill="#333">scan speed, MB/s (log)</text><text x="16" y="{}" transform="rotate(-90 16 {})" text-anchor="middle" fill="#333">string ratio</text>"##,
        w / 2.0,
        h - 16.0,
        h / 2.0,
        h / 2.0
    );
    for (ds, c, sx, sy) in points {
        let color = colors[codecs.iter().position(|k| k == c).unwrap_or(0) % colors.len()];
        let shape = shapes[datasets.iter().position(|d| d == ds).unwrap_or(0) % shapes.len()];
        let (cx, cy) = (x(*sx), y(*sy));
        let _ = match shape {
            "circle" => write!(s, r#"<circle cx="{cx}" cy="{cy}" r="6" fill="{color}"/>"#),
            "square" => write!(
                s,
                r#"<rect x="{}" y="{}" width="11" height="11" fill="{color}"/>"#,
                cx - 5.5,
                cy - 5.5
            ),
            _ => write!(
                s,
                r#"<polygon points="{},{} {},{} {},{}" fill="{color}"/>"#,
                cx,
                cy - 7.0,
                cx - 6.0,
                cy + 5.0,
                cx + 6.0,
                cy + 5.0
            ),
        };
    }
    for (i, c) in codecs.iter().enumerate() {
        let ly = 20.0 + i as f64 * 16.0;
        let _ = write!(
            s,
            r##"<rect x="{}" y="{}" width="10" height="10" fill="{}"/><text x="{}" y="{}" fill="#333">{c}</text>"##,
            w - 230.0,
            ly - 9.0,
            colors[i % colors.len()],
            w - 214.0,
            ly
        );
    }
    for (i, d) in datasets.iter().enumerate() {
        let _ = write!(
            s,
            r##"<text x="{}" y="{}" fill="#333">{} = {d}</text>"##,
            pad + 10.0,
            20.0 + i as f64 * 16.0,
            ["●", "■", "▲"][i % 3]
        );
    }
    s.push_str("</svg>\n");
    s
}
