//! Code schemes: how symbols are mapped to codes on the wire.

use core::fmt;

use crate::{Error, Result};

/// The code width and layout used by a dictionary and its frames.
///
/// The format reserves several schemes so that wide-code variants (thousands of
/// pretrained symbols) can be added without a format break. Only
/// [`Scheme::U8`] is implemented in this version; the others are rejected with
/// [`Error::UnsupportedScheme`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
#[repr(u8)]
pub enum Scheme {
    /// FSST-compatible: up to 255 symbols of 1 to 8 bytes, one-byte codes,
    /// code 255 is an escape followed by a literal byte.
    U8 = 0,
    /// Reserved: fixed two-byte codes, codes 0..=255 are implicit literals.
    U16 = 1,
    /// Reserved: hot symbols get one-byte codes, the rest two-byte codes.
    Hybrid8x16 = 2,
}

impl Scheme {
    /// Parses a scheme from its wire value, rejecting unimplemented schemes.
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(Scheme::U8),
            _ => Err(Error::UnsupportedScheme(v)),
        }
    }

    /// The wire value of this scheme.
    #[must_use]
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Maximum number of symbols a dictionary of this scheme may hold.
    #[must_use]
    pub fn max_symbols(self) -> usize {
        match self {
            Scheme::U8 => U8_MAX_SYMBOLS,
            Scheme::U16 => 65536 - 256,
            Scheme::Hybrid8x16 => 254 * 256,
        }
    }

    /// Maximum symbol length in bytes.
    #[must_use]
    pub fn max_symbol_len(self) -> usize {
        match self {
            Scheme::U8 => 8,
            Scheme::U16 | Scheme::Hybrid8x16 => 16,
        }
    }

    /// Upper bound on the encoded size of `n` input bytes.
    #[must_use]
    pub fn max_encoded_len(self, n: usize) -> usize {
        match self {
            // Worst case: every byte is escaped.
            Scheme::U8 | Scheme::Hybrid8x16 | Scheme::U16 => n.saturating_mul(2),
        }
    }
}

impl fmt::Display for Scheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Scheme::U8 => "u8",
            Scheme::U16 => "u16",
            Scheme::Hybrid8x16 => "hybrid8x16",
        })
    }
}

/// Number of symbol codes in the U8 scheme (code 255 is the escape).
pub(crate) const U8_MAX_SYMBOLS: usize = 255;
/// The U8 escape code.
pub(crate) const U8_ESCAPE: u8 = 255;
