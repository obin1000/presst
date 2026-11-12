//! Training corpora: records, sampling and the train/validation split.

use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use xxhash_rust::xxh3::Xxh3;

/// How input bytes are cut into training records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    /// Fixed-size windows; for bulk text such as logs or documents.
    Windows(usize),
    /// One record per line (without the newline); for string columns such as
    /// URLs or comments, where each value is compressed on its own.
    Lines,
}

impl Default for Split {
    fn default() -> Self {
        Split::Windows(4096)
    }
}

/// A training corpus: a list of records (byte strings).
///
/// Records are the unit of sampling and of the train/validation split, and the
/// encoder never matches across record boundaries during training, mirroring
/// how the data is compressed later.
#[derive(Debug, Clone, Default)]
pub struct Corpus {
    records: Vec<Vec<u8>>,
}

impl Corpus {
    /// Creates an empty corpus.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a corpus from records.
    #[must_use]
    pub fn from_records(records: Vec<Vec<u8>>) -> Self {
        Self { records }
    }

    /// Adds `data` (for example a file's contents), cut into records.
    pub fn add(&mut self, data: &[u8], split: Split) {
        match split {
            Split::Windows(size) => {
                self.records.extend(data.chunks(size.max(1)).map(<[u8]>::to_vec));
            }
            Split::Lines => {
                for line in data.split(|&b| b == b'\n') {
                    let line = line.strip_suffix(b"\r").unwrap_or(line);
                    if !line.is_empty() {
                        self.records.push(line.to_vec());
                    }
                }
            }
        }
    }

    /// The records.
    #[must_use]
    pub fn records(&self) -> &[Vec<u8>] {
        &self.records
    }

    /// Number of records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the corpus has no records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Total size in bytes.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.records.iter().map(Vec::len).sum()
    }

    /// A hash identifying the corpus content (record boundaries included).
    #[must_use]
    pub fn content_hash(&self) -> u64 {
        let mut h = Xxh3::new();
        for r in &self.records {
            h.update(&(r.len() as u64).to_le_bytes());
            h.update(r);
        }
        h.digest()
    }

    /// Shuffles the records deterministically and splits them into a training
    /// sample of about `sample_bytes` and a validation set of about
    /// `validation_bytes`, which do not overlap.
    pub(crate) fn sample(&self, seed: u64, sample_bytes: usize, validation_bytes: usize) -> Sample<'_> {
        let mut order: Vec<usize> = (0..self.records.len()).collect();
        order.shuffle(&mut ChaCha8Rng::seed_from_u64(seed));
        let mut validation = Vec::new();
        let mut train = Vec::new();
        let (mut vb, mut tb) = (0, 0);
        // Fill validation first so small corpora still get a validation set,
        // but never hold out more than a fifth of the data for it.
        let validation_bytes = validation_bytes.min(self.total_bytes() / 5);
        let max_validation = self.records.len() / 5;
        for &i in &order {
            let r = self.records[i].as_slice();
            if vb < validation_bytes && validation.len() < max_validation {
                vb += r.len();
                validation.push(r);
            } else if tb < sample_bytes {
                tb += r.len();
                train.push(r);
            } else {
                break;
            }
        }
        if validation.is_empty() {
            validation.clone_from(&train);
        }
        Sample { train, validation }
    }
}

/// A training sample and a disjoint validation set.
#[derive(Debug, Clone)]
pub(crate) struct Sample<'a> {
    pub(crate) train: Vec<&'a [u8]>,
    pub(crate) validation: Vec<&'a [u8]>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_modes() {
        let mut c = Corpus::new();
        c.add(b"ab\r\ncd\n\nef", Split::Lines);
        assert_eq!(c.records(), [b"ab".to_vec(), b"cd".to_vec(), b"ef".to_vec()]);
        let mut w = Corpus::new();
        w.add(&[7u8; 10], Split::Windows(4));
        assert_eq!(w.records().iter().map(Vec::len).collect::<Vec<_>>(), [4, 4, 2]);
    }

    #[test]
    fn sample_is_deterministic_and_disjoint() {
        let c = Corpus::from_records((0..1000u32).map(|i| i.to_le_bytes().to_vec()).collect());
        let a = c.sample(1, 400, 400);
        let b = c.sample(1, 400, 400);
        assert_eq!(a.train, b.train);
        assert_eq!(a.validation.len(), 100); // 400 bytes of 4-byte records, below the 20% cap
        assert!(a.train.iter().all(|t| !a.validation.contains(t)));
        assert_ne!(c.sample(2, 400, 400).train, a.train);
    }
}
