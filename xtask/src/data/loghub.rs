//! Loghub system logs: each log line is one value. Logs are chronological, so
//! the first 25% of lines train and the rest is test data (level L2, past ->
//! future). Test lines are labelled with the time elapsed since the last
//! training line, parsed from each system's own timestamp format.

use std::path::Path;

use anyhow::Result;

use super::{Column, Meta, fetch, read_all, write_table};

/// Share of each log (by lines, in time order) used for training.
const TRAIN_SHARE: f64 = 0.25;

pub fn run() -> Result<()> {
    let mut total = 0;
    for path in fetch("loghub")? {
        let file = path.file_name().expect("file name").to_string_lossy().into_owned();
        let system = system_name(&file);
        eprintln!("loghub {system}");
        let lines = read_lines(&path)?;
        let (lines, dropped): (Vec<Vec<u8>>, usize) = {
            let n = lines.len();
            let kept: Vec<Vec<u8>> = lines.into_iter().filter(|l| !l.is_empty()).collect();
            let d = n - kept.len();
            (kept, d)
        };
        let split = (lines.len() as f64 * TRAIN_SHARE) as usize;
        let times = timestamps(&system, &lines);
        let train_end = times[..split].iter().rev().find(|&&t| t != i64::MIN).copied();
        let test_time: Vec<i64> = times[split..]
            .iter()
            .map(|&t| match (t, train_end) {
                (t, Some(end)) if t != i64::MIN => t - end,
                _ => i64::MIN,
            })
            .collect();
        let parsed = test_time.iter().filter(|&&t| t != i64::MIN).count();
        let has_time = parsed * 2 > test_time.len();
        if let (Some(&first), Some(&last)) = (
            test_time.iter().find(|&&t| t != i64::MIN),
            test_time.iter().rev().find(|&&t| t != i64::MIN),
        ) {
            eprintln!(
                "  test period: +{:.1} h to +{:.1} days after training ({:.0}% of lines have timestamps)",
                first as f64 / 3600.0,
                last as f64 / 86400.0,
                100.0 * parsed as f64 / test_time.len().max(1) as f64
            );
        }
        let mut lines = lines;
        let test = lines.split_off(split);
        let meta = Meta {
            dataset: "loghub".into(),
            table: system.clone(),
            level: "L2".into(),
            split: format!(
                "chronological: first {:.0}% of lines train, the rest test",
                TRAIN_SHARE * 100.0
            ),
            train_sources: vec![format!("{file}#lines[0..{split})")],
            test_sources: vec![format!("{file}#lines[{split}..)")],
            columns: Vec::new(),
        };
        let col = Column {
            name: "line".into(),
            train: lines,
            test,
            test_time: has_time.then_some(test_time),
            dropped,
        };
        total += write_table(meta, vec![col])?;
    }
    eprintln!("loghub: {total} columns");
    Ok(())
}

fn system_name(file: &str) -> String {
    let stem = file.split('.').next().unwrap_or(file);
    match stem {
        "SSH" => "OpenSSH".into(),
        "HDFS_v1" => "HDFS".into(),
        s => s.into(),
    }
}

/// All `.log` files of an archive, sorted by path, split into lines.
fn read_lines(path: &Path) -> Result<Vec<Vec<u8>>> {
    let name = path.to_string_lossy();
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    if name.ends_with(".zip") {
        let mut zip = zip::ZipArchive::new(std::fs::File::open(path)?)?;
        for i in 0..zip.len() {
            let entry = zip.by_index(i)?;
            let n = entry.name()?.into_owned();
            if n.ends_with(".log") && !n.contains("preprocessed") {
                files.push((n, read_all(entry)?));
            }
        }
    } else {
        let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(std::fs::File::open(path)?));
        for entry in tar.entries()? {
            let entry = entry?;
            let n = entry.path()?.to_string_lossy().into_owned();
            if n.ends_with(".log") {
                files.push((n, read_all(entry)?));
            }
        }
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));
    let mut lines = Vec::new();
    for (_, data) in files {
        for line in data.split(|&b| b == b'\n') {
            lines.push(line.strip_suffix(b"\r").unwrap_or(line).to_vec());
        }
    }
    Ok(lines)
}

/// Seconds since an arbitrary epoch per line, or `i64::MIN` if unparsable.
fn timestamps(system: &str, lines: &[Vec<u8>]) -> Vec<i64> {
    let mut year_state = YearState::default();
    lines
        .iter()
        .map(|l| {
            let s = std::str::from_utf8(&l[..l.len().min(120)]).unwrap_or("");
            let t = match system {
                "BGL" | "HPC" => epoch_token(s),
                "HDFS" => hdfs(s),
                "Spark" => spark(s),
                "Apache" => apache(s),
                "Linux" | "Mac" | "OpenSSH" => syslog(s, &mut year_state),
                "Proxifier" => proxifier(s, &mut year_state),
                _ => iso(s),
            };
            t.unwrap_or(i64::MIN)
        })
        .collect()
}

/// Days since 1970-01-01 (proleptic Gregorian).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn secs(y: i64, mo: i64, d: i64, h: i64, mi: i64, s: i64) -> Option<i64> {
    ((1..=12).contains(&mo) && (1..=31).contains(&d) && h < 24 && mi < 60 && s < 61)
        .then(|| days_from_civil(y, mo, d) * 86400 + h * 3600 + mi * 60 + s)
}

fn num(s: &str) -> Option<i64> {
    (!s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
        .then(|| s.parse().ok())
        .flatten()
}

fn month(s: &str) -> Option<i64> {
    const M: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    M.iter().position(|&m| m == s).map(|i| i as i64 + 1)
}

fn hms(s: &str) -> Option<(i64, i64, i64)> {
    let mut it = s.split(':');
    Some((num(it.next()?)?, num(it.next()?)?, num(it.next()?.get(..2)?)?))
}

/// First whitespace token that is a 10-digit Unix timestamp (BGL, HPC).
fn epoch_token(s: &str) -> Option<i64> {
    s.split_whitespace().find(|t| t.len() == 10).and_then(num)
}

/// `081109 203518 ...` (yymmdd HHMMSS).
fn hdfs(s: &str) -> Option<i64> {
    let mut it = s.split_whitespace();
    let (d, t) = (it.next()?, it.next()?);
    if d.len() != 6 || t.len() != 6 {
        return None;
    }
    secs(
        2000 + num(&d[0..2])?,
        num(&d[2..4])?,
        num(&d[4..6])?,
        num(&t[0..2])?,
        num(&t[2..4])?,
        num(&t[4..6])?,
    )
}

/// `17/06/09 20:10:40 ...` (yy/MM/dd HH:mm:ss).
fn spark(s: &str) -> Option<i64> {
    let mut it = s.split_whitespace();
    let d: Vec<&str> = it.next()?.split('/').collect();
    let (h, mi, se) = hms(it.next()?)?;
    if d.len() != 3 {
        return None;
    }
    secs(2000 + num(d[0])?, num(d[1])?, num(d[2])?, h, mi, se)
}

/// `[Thu Jun 09 06:07:04 2005] ...`
fn apache(s: &str) -> Option<i64> {
    let inner = s.strip_prefix('[')?.split(']').next()?;
    let p: Vec<&str> = inner.split_whitespace().collect();
    if p.len() != 5 {
        return None;
    }
    let (h, mi, se) = hms(p[3])?;
    secs(num(p[4])?, month(p[1])?, num(p[2])?, h, mi, se)
}

/// `YYYY-MM-DD HH:MM:SS` anywhere near the start (Zookeeper, Hadoop, OpenStack…).
fn iso(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    (0..b.len().saturating_sub(18)).find_map(|i| {
        let c = &s.get(i..i + 19)?;
        let cb = c.as_bytes();
        if cb[4] != b'-' || cb[7] != b'-' || !(cb[10] == b' ' || cb[10] == b'T') || cb[13] != b':' {
            return None;
        }
        let (h, mi, se) = hms(&c[11..19])?;
        secs(num(&c[0..4])?, num(&c[5..7])?, num(&c[8..10])?, h, mi, se)
    })
}

/// Tracks the year for formats without one: it increments whenever the month
/// goes backwards (logs are chronological).
#[derive(Default)]
struct YearState {
    year: i64,
    last_month: i64,
}

impl YearState {
    fn year_for(&mut self, month: i64) -> i64 {
        if self.last_month != 0 && month < self.last_month {
            self.year += 1;
        }
        self.last_month = month;
        2000 + self.year
    }
}

/// `Jun  9 06:06:20 host ...` (no year).
fn syslog(s: &str, st: &mut YearState) -> Option<i64> {
    let mut it = s.split_whitespace();
    let mo = month(it.next()?)?;
    let d = num(it.next()?)?;
    let (h, mi, se) = hms(it.next()?)?;
    secs(st.year_for(mo), mo, d, h, mi, se)
}

/// `[10.30 16:49:06] ...` (MM.DD, no year).
fn proxifier(s: &str, st: &mut YearState) -> Option<i64> {
    let inner = s.strip_prefix('[')?.split(']').next()?;
    let (date, time) = inner.split_once(' ')?;
    let (mo, d) = date.split_once('.')?;
    let (mo, d) = (num(mo)?, num(d)?);
    let (h, mi, se) = hms(time)?;
    secs(st.year_for(mo), mo, d, h, mi, se)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parsers() {
        let day = 86400;
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(2000, 3, 1), 11_017);
        assert_eq!(epoch_token("- 1117838570 2005.06.03 R02"), Some(1_117_838_570));
        let a = hdfs("081109 203518 143 INFO").unwrap();
        assert_eq!(hdfs("081110 203518 1").unwrap() - a, day);
        let a = apache("[Thu Jun 09 06:07:04 2005] [notice]").unwrap();
        assert_eq!(apache("[Fri Jun 10 06:07:04 2005] x").unwrap() - a, day);
        let a = iso("nova-api.log.1 2017-05-16 00:00:00.008 25746").unwrap();
        assert_eq!(iso("2017-05-17 00:00:00,536 - INFO").unwrap() - a, day);
        assert_eq!(
            spark("17/06/10 00:00:00 INFO x").unwrap() - spark("17/06/09 00:00:00 INFO").unwrap(),
            day
        );
        let mut st = YearState::default();
        let dec = syslog("Dec 31 23:00:00 host a", &mut st).unwrap();
        let jan = syslog("Jan  1 01:00:00 host b", &mut st).unwrap();
        assert_eq!(jan - dec, 2 * 3600, "year rolls over");
        let mut st = YearState::default();
        assert!(proxifier("[10.30 16:49:06] chrome.exe", &mut st).is_some());
        assert_eq!(apache("garbage"), None);
    }
}
