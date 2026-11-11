//! Offline trainer for presst dictionaries.
//!
//! Two modes:
//!
//! * [`Mode::Fsst`] is a faithful port of FSST's per-block training: a 16 KiB
//!   sample and five generations with a growing sample fraction. It is the
//!   baseline that shows what per-block training would achieve.
//! * [`Mode::Extended`] spends far more compute, which is the point of
//!   pretraining: a large sample, many generations with early stopping on a
//!   held-out validation set, a final refinement round, and local search that
//!   swaps symbols when that shrinks the validation output.
//!
//! Training is deterministic: the same corpus, options and seed produce the
//! same dictionary (and [`DictId`](presst::DictId)) on any number of threads.
//!
//! ```
//! use presst_train::{train, Corpus, Split, TrainOptions};
//!
//! let mut corpus = Corpus::new();
//! corpus.add(b"GET /index.html\nGET /about.html\nPOST /login\n".repeat(50).as_slice(), Split::Lines);
//! let report = train(&corpus, &TrainOptions::extended())?;
//! assert!(report.validation_ratio > 2.0);
//! # Ok::<(), presst_train::TrainError>(())
//! ```

mod corpus;

use std::collections::HashMap;
use std::fmt;

use presst::{Compressor, Dictionary, DictionaryBuilder, Metadata, Scheme};
use rayon::prelude::*;

use crate::corpus::Sample;
pub use crate::corpus::{Corpus, Split};

/// Training strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// FSST's own algorithm: 16 KiB sample, 5 generations.
    Fsst,
    /// Large sample, many generations, validation-based selection and local search.
    Extended,
}

/// Training options. Start from [`TrainOptions::extended`] or [`TrainOptions::fsst`].
#[derive(Debug, Clone)]
pub struct TrainOptions {
    /// Training strategy.
    pub mode: Mode,
    /// Code scheme of the dictionary.
    pub scheme: Scheme,
    /// Seed for sampling; the result is deterministic per seed.
    pub seed: u64,
    /// Bytes of training sample.
    pub sample_bytes: usize,
    /// Bytes held out to select between generations and judge local search moves.
    pub validation_bytes: usize,
    /// Maximum number of generations ([`Mode::Extended`] only).
    pub max_generations: usize,
    /// Stop after this many generations without validation improvement.
    pub patience: usize,
    /// Number of symbol swaps to try after the generations (0 disables).
    pub local_search_steps: usize,
    /// Number of threads; `None` uses all cores.
    pub threads: Option<usize>,
    /// Name stored in the dictionary metadata.
    pub name: Option<String>,
}

impl TrainOptions {
    /// Options for [`Mode::Extended`] (the default).
    #[must_use]
    pub fn extended() -> Self {
        Self {
            mode: Mode::Extended,
            scheme: Scheme::U8,
            seed: 0x5EED,
            sample_bytes: 16 << 20,
            validation_bytes: 4 << 20,
            max_generations: 50,
            patience: 6,
            local_search_steps: 300,
            threads: None,
            name: None,
        }
    }

    /// Options for [`Mode::Fsst`]: FSST's sample size and five generations.
    #[must_use]
    pub fn fsst() -> Self {
        Self {
            mode: Mode::Fsst,
            sample_bytes: 16 << 10,
            validation_bytes: 4 << 20,
            max_generations: 5,
            local_search_steps: 0,
            ..Self::extended()
        }
    }
}

impl Default for TrainOptions {
    fn default() -> Self {
        Self::extended()
    }
}

/// Validation result of one generation.
#[derive(Debug, Clone, Copy)]
pub struct GenerationStats {
    /// Generation number, starting at 1.
    pub generation: usize,
    /// Symbols in the table.
    pub symbols: usize,
    /// Compression ratio (input / encoded) on the validation set.
    pub validation_ratio: f64,
}

/// The trained dictionary plus statistics.
#[derive(Debug, Clone)]
pub struct TrainReport {
    /// The trained dictionary, with metadata filled in.
    pub dictionary: Dictionary,
    /// Per-generation validation results.
    pub generations: Vec<GenerationStats>,
    /// Number of accepted local search swaps.
    pub local_search_accepted: usize,
    /// Final compression ratio on the validation set.
    pub validation_ratio: f64,
    /// Bytes in the training sample.
    pub sample_bytes: usize,
    /// Bytes in the validation set.
    pub validation_bytes: usize,
}

/// Training errors.
#[derive(Debug)]
#[non_exhaustive]
pub enum TrainError {
    /// The corpus has no data.
    EmptyCorpus,
    /// The thread pool could not be created.
    ThreadPool(rayon::ThreadPoolBuildError),
    /// The core library rejected the options or dictionary.
    Presst(presst::Error),
}

impl fmt::Display for TrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrainError::EmptyCorpus => f.write_str("the training corpus is empty"),
            TrainError::ThreadPool(e) => write!(f, "could not create thread pool: {e}"),
            TrainError::Presst(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for TrainError {}

impl From<presst::Error> for TrainError {
    fn from(e: presst::Error) -> Self {
        TrainError::Presst(e)
    }
}

/// Trains a dictionary on `corpus`.
pub fn train(corpus: &Corpus, opts: &TrainOptions) -> Result<TrainReport, TrainError> {
    if corpus.total_bytes() == 0 {
        return Err(TrainError::EmptyCorpus);
    }
    let run = || train_inner(corpus, opts);
    match opts.threads {
        Some(n) => rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map_err(TrainError::ThreadPool)?
            .install(run),
        None => run(),
    }
}

fn train_inner(corpus: &Corpus, opts: &TrainOptions) -> Result<TrainReport, TrainError> {
    let sample = corpus.sample(opts.seed, opts.sample_bytes, opts.validation_bytes);
    let (dict, generations, accepted) = match opts.mode {
        Mode::Fsst => {
            let (d, g) = fsst_generations(&sample, opts.scheme)?;
            (d, g, 0)
        }
        Mode::Extended => extended(&sample, opts)?,
    };
    let validation_ratio = ratio(&dict, &sample.validation)?;
    let dictionary = dict.with_metadata(Metadata {
        name: opts.name.clone(),
        trainer: Some(format!(
            "presst-train {} ({})",
            env!("CARGO_PKG_VERSION"),
            match opts.mode {
                Mode::Fsst => "fsst",
                Mode::Extended => "extended",
            }
        )),
        seed: Some(opts.seed),
        corpus_hash: Some(corpus.content_hash()),
    });
    Ok(TrainReport {
        dictionary,
        generations,
        local_search_accepted: accepted,
        validation_ratio,
        sample_bytes: sample.train.iter().map(|r| r.len()).sum(),
        validation_bytes: sample.validation.iter().map(|r| r.len()).sum(),
    })
}

/// Compresses `records` one by one (no matches across records) and returns
/// `(input bytes, encoded bytes)`.
pub fn evaluate(dict: &Dictionary, records: &[&[u8]]) -> Result<(u64, u64), presst::Error> {
    let comp = Compressor::new(dict)?;
    Ok(records
        .par_iter()
        .with_min_len(64)
        .map_init(Vec::new, |buf, r| {
            buf.clear();
            (r.len() as u64, comp.compress_block_into(r, buf) as u64)
        })
        .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1)))
}

fn ratio(dict: &Dictionary, records: &[&[u8]]) -> Result<f64, presst::Error> {
    let (input, encoded) = evaluate(dict, records)?;
    Ok(if encoded == 0 {
        1.0
    } else {
        input as f64 / encoded as f64
    })
}

/// FSST's schedule: 5 generations over a growing fraction of the sample
/// (8/128 up to 128/128). The last round only keeps symbols that are used.
fn fsst_generations(sample: &Sample<'_>, scheme: Scheme) -> Result<(Dictionary, Vec<GenerationStats>), TrainError> {
    let mut dict = DictionaryBuilder::new(scheme)?.build();
    let mut stats = Vec::new();
    for (g, frac) in [8usize, 38, 68, 98, 128].into_iter().enumerate() {
        let n = (sample.train.len() * frac).div_ceil(128).max(1).min(sample.train.len());
        let counts = Counts::collect(&dict, &sample.train[..n])?;
        let final_round = frac == 128;
        let min_count = if final_round { 0 } else { (5 * frac as u64) / 128 };
        dict = build_table(scheme, &counts.candidates(&dict, !final_round, min_count))?;
        stats.push(GenerationStats {
            generation: g + 1,
            symbols: dict.len(),
            validation_ratio: ratio(&dict, &sample.validation)?,
        });
    }
    Ok((dict, stats))
}

/// Many generations over the full sample, keeping the best table on the
/// validation set, followed by a refinement round and local search.
fn extended(sample: &Sample<'_>, opts: &TrainOptions) -> Result<(Dictionary, Vec<GenerationStats>, usize), TrainError> {
    let scheme = opts.scheme;
    let mut dict = DictionaryBuilder::new(scheme)?.build();
    let mut best = (dict.clone(), encoded_size(&dict, &sample.validation)?);
    let mut stats = Vec::new();
    let mut since_best = 0;
    for g in 1..=opts.max_generations.max(1) {
        let counts = Counts::collect(&dict, &sample.train)?;
        dict = build_table(scheme, &counts.candidates(&dict, true, 2))?;
        let size = encoded_size(&dict, &sample.validation)?;
        stats.push(GenerationStats {
            generation: g,
            symbols: dict.len(),
            validation_ratio: ratio_of(sample, size),
        });
        if size < best.1 {
            best = (dict.clone(), size);
            since_best = 0;
        } else {
            since_best += 1;
            if since_best >= opts.patience {
                break;
            }
        }
    }

    // Refinement: rebuild from the symbols the best table actually uses.
    let counts = Counts::collect(&best.0, &sample.train)?;
    let refined = build_table(scheme, &counts.candidates(&best.0, false, 0))?;
    let refined_size = encoded_size(&refined, &sample.validation)?;
    if refined_size < best.1 {
        best = (refined, refined_size);
    }

    let (dict, accepted) = local_search(best, sample, opts)?;
    Ok((dict, stats, accepted))
}

fn ratio_of(sample: &Sample<'_>, encoded: u64) -> f64 {
    let input: usize = sample.validation.iter().map(|r| r.len()).sum();
    if encoded == 0 {
        1.0
    } else {
        input as f64 / encoded as f64
    }
}

fn encoded_size(dict: &Dictionary, records: &[&[u8]]) -> Result<u64, presst::Error> {
    Ok(evaluate(dict, records)?.1)
}

/// Tries to replace the least useful symbols with the best unused candidates,
/// keeping a swap only if the validation output shrinks.
fn local_search(
    (mut dict, mut size): (Dictionary, u64),
    sample: &Sample<'_>,
    opts: &TrainOptions,
) -> Result<(Dictionary, usize), TrainError> {
    if opts.local_search_steps == 0 {
        return Ok((dict, 0));
    }
    let counts = Counts::collect(&dict, &sample.train)?;
    let mut symbols: Vec<Vec<u8>> = dict.symbols().map(<[u8]>::to_vec).collect();
    // Weakest first: the symbols that save the least on the training sample.
    let usage = counts.symbol_usage(&dict);
    let mut weakest: Vec<usize> = (0..symbols.len()).collect();
    weakest.sort_by_key(|&i| (usage[i], i));
    let pool: Vec<Vec<u8>> = counts
        .candidates(&dict, true, 2)
        .into_iter()
        .map(|(_, s)| s)
        .filter(|s| !symbols.contains(s))
        .take(opts.local_search_steps)
        .collect();

    let mut accepted = 0;
    let mut w = 0;
    for cand in pool {
        let full = symbols.len() >= opts.scheme.max_symbols();
        let mut trial = symbols.clone();
        if full {
            let Some(&victim) = weakest.get(w) else { break };
            trial.remove(victim);
            trial.push(cand);
        } else {
            trial.push(cand);
        }
        let Ok(trial_dict) = Dictionary::from_symbols(opts.scheme, &trial) else {
            continue;
        };
        let trial_size = encoded_size(&trial_dict, &sample.validation)?;
        if trial_size < size {
            accepted += 1;
            size = trial_size;
            dict = trial_dict;
            if full {
                let victim = weakest[w];
                // Indices after the removed symbol shift down by one.
                weakest.remove(w);
                for i in &mut weakest {
                    if *i > victim {
                        *i -= 1;
                    }
                }
                weakest.push(trial.len() - 1);
            }
            symbols = trial;
        } else if full {
            w += 1;
        }
    }
    Ok((dict, accepted))
}

/// Picks the highest-gain candidates that satisfy the scheme's constraints.
fn build_table(scheme: Scheme, candidates: &[(u64, Vec<u8>)]) -> Result<Dictionary, TrainError> {
    let mut b = DictionaryBuilder::new(scheme)?;
    for (_, sym) in candidates {
        if b.is_full() {
            break;
        }
        b.push_lossy(sym);
    }
    Ok(b.build())
}

/// Token ids: dictionary codes of multi-byte symbols are `0..255`; every single
/// byte `b` (whether a 1-byte symbol or an escape) is `256 + b`.
const IDS: usize = 512;

/// Symbol and symbol-pair frequencies from parsing a sample with a table.
struct Counts {
    single: Vec<u64>,
    pairs: Vec<u64>,
}

impl Counts {
    fn new() -> Self {
        Self {
            single: vec![0; IDS],
            pairs: vec![0; IDS * IDS],
        }
    }

    fn merge(mut self, other: Self) -> Self {
        for (a, b) in self.single.iter_mut().zip(other.single) {
            *a += b;
        }
        for (a, b) in self.pairs.iter_mut().zip(other.pairs) {
            *a += b;
        }
        self
    }

    /// Parses every record with `dict` and counts tokens and adjacent pairs,
    /// like FSST's `compressCount`.
    fn collect(dict: &Dictionary, records: &[&[u8]]) -> Result<Self, presst::Error> {
        let comp = Compressor::new(dict)?;
        let ids = TokenIds::new(dict);
        Ok(records
            .par_iter()
            .with_min_len(256)
            .fold(
                || (Counts::new(), Vec::new()),
                |(mut counts, mut codes), record| {
                    codes.clear();
                    comp.compress_block_into(record, &mut codes);
                    counts.add_record(&ids, &codes);
                    (counts, codes)
                },
            )
            .map(|(c, _)| c)
            .reduce(Counts::new, Counts::merge))
    }

    fn add_record(&mut self, ids: &TokenIds, codes: &[u8]) {
        let mut prev: Option<usize> = None;
        let mut i = 0;
        while i < codes.len() {
            let (id, first) = if codes[i] == 255 {
                i += 2;
                let b = codes[i - 1] as usize;
                (256 + b, 256 + b)
            } else {
                i += 1;
                (ids.id[codes[i - 1] as usize], ids.first[codes[i - 1] as usize])
            };
            self.single[id] += 1;
            if id != first {
                // Also count the first byte alone, as an alternative to the
                // longer symbol.
                self.single[first] += 1;
            }
            if let Some(p) = prev {
                self.pairs[p * IDS + id] += 1;
                if id != first {
                    self.pairs[p * IDS + first] += 1;
                }
            }
            prev = Some(id);
        }
    }

    /// Candidate symbols with FSST's gain heuristic, highest gain first.
    ///
    /// Gain is `count * length`; single bytes get 8x, which lowers the escape
    /// rate. With `extend`, concatenations of adjacent tokens (truncated to 8
    /// bytes) are candidates too. Ties are broken deterministically.
    fn candidates(&self, dict: &Dictionary, extend: bool, min_count: u64) -> Vec<(u64, Vec<u8>)> {
        let mut gains: HashMap<Vec<u8>, u64> = HashMap::new();
        for id1 in 0..IDS {
            let c1 = self.single[id1];
            if c1 == 0 || c1 < min_count {
                continue;
            }
            let s1 = TokenIds::bytes(dict, id1);
            let weight = if s1.len() == 1 { 8 } else { 1 };
            *gains.entry(s1.clone()).or_default() += weight * c1 * s1.len() as u64;
            if !extend {
                continue;
            }
            for id2 in 0..IDS {
                let c2 = self.pairs[id1 * IDS + id2];
                if c2 == 0 || c2 < min_count {
                    continue;
                }
                let mut s3 = s1.clone();
                s3.extend_from_slice(&TokenIds::bytes(dict, id2));
                s3.truncate(dict.scheme().max_symbol_len());
                *gains.entry(s3.clone()).or_default() += c2 * s3.len() as u64;
            }
        }
        let mut out: Vec<(u64, Vec<u8>)> = gains.into_iter().map(|(s, g)| (g, s)).collect();
        out.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(b.1.len().cmp(&a.1.len())).then(a.1.cmp(&b.1)));
        out
    }

    /// Bytes each dictionary symbol covered in the sample (`count * len`).
    fn symbol_usage(&self, dict: &Dictionary) -> Vec<u64> {
        let ids = TokenIds::new(dict);
        (0..dict.len())
            .map(|c| self.single[ids.id[c]] * dict.symbol(c).len() as u64)
            .collect()
    }
}

/// Maps dictionary codes to token ids.
struct TokenIds {
    /// Token id per code.
    id: [usize; 256],
    /// Token id of the code's first byte.
    first: [usize; 256],
}

impl TokenIds {
    fn new(dict: &Dictionary) -> Self {
        let mut id = [0; 256];
        let mut first = [0; 256];
        for (c, s) in dict.symbols().enumerate() {
            first[c] = 256 + s[0] as usize;
            id[c] = if s.len() == 1 { first[c] } else { c };
        }
        Self { id, first }
    }

    fn bytes(dict: &Dictionary, id: usize) -> Vec<u8> {
        if id >= 256 {
            vec![(id - 256) as u8]
        } else {
            dict.symbol(id).to_vec()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write as _;

    fn corpus() -> Corpus {
        let mut c = Corpus::new();
        let text = "the quick brown fox jumps over the lazy dog. pack my box with five dozen liquor jugs. \
                    how vexingly quick daft zebras jump! the five boxing wizards jump quickly.\n";
        let mut data = String::new();
        for i in 0..2000 {
            let _ = write!(data, "{i:05} {text}");
        }
        c.add(data.as_bytes(), Split::Windows(512));
        c
    }

    fn small(mut opts: TrainOptions) -> TrainOptions {
        opts.sample_bytes = 256 << 10;
        opts.validation_bytes = 64 << 10;
        opts.local_search_steps = 40;
        opts
    }

    #[test]
    fn fsst_mode_compresses() {
        let report = train(&corpus(), &TrainOptions::fsst()).unwrap();
        assert_eq!(report.generations.len(), 5);
        assert!(report.validation_ratio > 2.0, "ratio {}", report.validation_ratio);
    }

    #[test]
    fn extended_beats_fsst() {
        let c = corpus();
        let fsst = train(&c, &TrainOptions::fsst()).unwrap();
        let ext = train(&c, &small(TrainOptions::extended())).unwrap();
        assert!(
            ext.validation_ratio >= fsst.validation_ratio,
            "{} < {}",
            ext.validation_ratio,
            fsst.validation_ratio
        );
        assert_eq!(ext.dictionary.len(), 255);
    }

    #[test]
    fn deterministic_across_thread_counts() {
        let c = corpus();
        let one = train(
            &c,
            &TrainOptions {
                threads: Some(1),
                ..small(TrainOptions::extended())
            },
        )
        .unwrap();
        let four = train(
            &c,
            &TrainOptions {
                threads: Some(4),
                ..small(TrainOptions::extended())
            },
        )
        .unwrap();
        assert_eq!(one.dictionary.id(), four.dictionary.id());
        let other_seed = train(
            &c,
            &TrainOptions {
                seed: 99,
                ..small(TrainOptions::extended())
            },
        )
        .unwrap();
        assert_eq!(other_seed.dictionary.metadata().seed, Some(99));
    }

    #[test]
    fn empty_corpus() {
        assert!(matches!(
            train(&Corpus::new(), &TrainOptions::default()),
            Err(TrainError::EmptyCorpus)
        ));
    }
}
