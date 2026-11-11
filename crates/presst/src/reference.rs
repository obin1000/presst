//! Slow, obviously-correct reference implementations of the codecs.
//!
//! The optimized kernels must produce byte-identical output to these
//! functions; the test suite checks this with property tests. They are public
//! so the trainer and external tools can use them as an oracle, but they are
//! far too slow for production use.

use alloc::vec::Vec;

use crate::scheme::U8_ESCAPE;
use crate::{Dictionary, Error, Result, Scheme};

/// Encodes `input` with greedy longest-symbol matching.
///
/// At every position the longest symbol that matches is emitted as its code;
/// if no symbol matches, the byte is escaped. Because of the U8 prefix
/// constraint, this is exactly the parse the fast encoder produces.
///
/// # Panics
/// Panics if the dictionary does not use [`Scheme::U8`].
#[must_use]
pub fn encode_u8(dict: &Dictionary, input: &[u8]) -> Vec<u8> {
    assert_eq!(dict.scheme(), Scheme::U8);
    let mut out = Vec::with_capacity(input.len() * 2);
    let mut pos = 0;
    while pos < input.len() {
        let rest = &input[pos..];
        let best = dict
            .symbols()
            .enumerate()
            .filter(|(_, s)| rest.starts_with(s))
            .max_by_key(|(_, s)| s.len());
        if let Some((code, sym)) = best {
            out.push(code as u8);
            pos += sym.len();
        } else {
            out.push(U8_ESCAPE);
            out.push(rest[0]);
            pos += 1;
        }
    }
    out
}

/// Decodes a U8 code stream.
///
/// # Errors
/// Returns [`Error::InvalidCode`] for codes outside the dictionary and
/// [`Error::Truncated`] for an escape code at the end of the stream.
///
/// # Panics
/// Panics if the dictionary does not use [`Scheme::U8`].
pub fn decode_u8(dict: &Dictionary, codes: &[u8]) -> Result<Vec<u8>> {
    assert_eq!(dict.scheme(), Scheme::U8);
    let mut out = Vec::with_capacity(codes.len() * 3);
    let mut it = codes.iter();
    while let Some(&code) = it.next() {
        if code == U8_ESCAPE {
            out.push(*it.next().ok_or(Error::Truncated)?);
        } else if (code as usize) < dict.len() {
            out.extend_from_slice(dict.symbol(code as usize));
        } else {
            return Err(Error::InvalidCode);
        }
    }
    Ok(out)
}
