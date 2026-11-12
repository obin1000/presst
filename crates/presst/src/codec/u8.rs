//! The U8 (FSST-compatible) encode and decode kernels.
//!
//! Code stream: bytes `0..=254` are symbol codes, `255` is an escape followed
//! by one literal byte. This is the same stream FSST produces, so an FSST
//! decoder loaded with the same symbol table decodes presst U8 blocks.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use crate::scheme::U8_ESCAPE;
use crate::util::pack_le;
use crate::{Dictionary, Error, Result};

/// Entry in the 1- and 2-byte lookup tables.
///
/// Bits 0..8: code. Bits 8..12: bytes consumed (1 or 2). Bit 12: escape.
type ShortEntry = u16;

const SHORT_LEN_SHIFT: u32 = 8;
const SHORT_ESC_BIT: u16 = 1 << 12;
const ESCAPE_ENTRY: ShortEntry = U8_ESCAPE as u16 | (1 << SHORT_LEN_SHIFT) | SHORT_ESC_BIT;

/// Hash table entry for symbols of 3 or more bytes.
#[derive(Clone, Copy, Default)]
#[repr(C, align(16))]
struct HashEntry {
    /// Symbol bytes, little-endian, zero padded.
    sym: u64,
    /// Number of high bits to ignore when comparing (64 - 8 * len).
    ignored_bits: u8,
    code: u8,
    /// Symbol length; 0 marks an empty slot.
    len: u8,
}

/// Hash multipliers tried per table size when searching for a collision-free
/// hash function. With up to 255 distinct keys, a table of 2^16 slots has no
/// collision for about 60% of random multipliers, so the chance that all
/// attempts fail is negligible (about 0.4^256).
const HASH_ATTEMPTS: u32 = 256;

/// The `i`-th candidate multiplier: a fixed pseudo-random odd sequence
/// (`SplitMix64`), so the chosen hash function is deterministic.
fn hash_multiplier(i: u32) -> u64 {
    let mut z = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(u64::from(i) + 1);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) | 1
}

/// Prepared encoder tables for a U8 dictionary.
pub(crate) struct U8Encoder {
    /// Indexed by the next two input bytes (little-endian).
    short: Box<[ShortEntry]>,
    /// Indexed by the next input byte; used when only one byte is left.
    byte: [ShortEntry; 256],
    /// Collision-free hash table over the 3-byte prefixes of long symbols.
    hash: Box<[HashEntry]>,
    hash_mul: u64,
    hash_shift: u32,
}

impl U8Encoder {
    pub(crate) fn new(dict: &Dictionary) -> Result<Self> {
        let mut byte = [ESCAPE_ENTRY; 256];
        let mut two: Vec<Option<u8>> = vec![None; 65536];
        let mut long: Vec<(u32, HashEntry)> = Vec::new();
        for (code, sym) in dict.symbols().enumerate() {
            let code8 = code as u8;
            match sym.len() {
                1 => byte[sym[0] as usize] = code8 as u16 | (1 << SHORT_LEN_SHIFT),
                2 => two[u16::from_le_bytes([sym[0], sym[1]]) as usize] = Some(code8),
                len => long.push((
                    prefix_key(pack_le(sym)),
                    HashEntry {
                        sym: pack_le(sym),
                        ignored_bits: (64 - 8 * len) as u8,
                        code: code8,
                        len: len as u8,
                    },
                )),
            }
        }
        let mut short = vec![0u16; 65536].into_boxed_slice();
        for (i, slot) in short.iter_mut().enumerate() {
            *slot = match two[i] {
                Some(code) => code as u16 | (2 << SHORT_LEN_SHIFT),
                None => byte[i & 0xFF],
            };
        }
        let (hash, hash_mul, hash_shift) = build_hash(&long)?;
        Ok(Self {
            short,
            byte,
            hash,
            hash_mul,
            hash_shift,
        })
    }

    #[inline(always)]
    fn slot(&self, word: u64) -> usize {
        (prefix_key(word) as u64)
            .wrapping_mul(self.hash_mul)
            .wrapping_shr(self.hash_shift) as usize
    }

    /// Appends the code stream for `input` to `out`.
    pub(crate) fn encode(&self, input: &[u8], out: &mut Vec<u8>) {
        let n = input.len();
        // Worst case every byte is escaped (2 bytes each), plus 1 byte of slack
        // for the unconditional 2-byte store.
        out.reserve(2 * n + 2);
        let start = out.len();
        let base = out.as_mut_ptr();
        let mut op = start;
        let mut ip = 0;

        // Fast path: at least 8 input bytes left, so every symbol fits.
        while ip + 8 <= n {
            // SAFETY: ip + 8 <= n, so the 8-byte read is inside `input`.
            let word = u64::from_le(unsafe { input.as_ptr().add(ip).cast::<u64>().read_unaligned() });
            // SAFETY: `op <= start + 2 * ip` (each input byte emits at most 2
            // bytes) and ip < n, so op + 2 <= start + 2n <= capacity.
            let (consumed, written) = unsafe { self.step(word, 8, base.add(op)) };
            ip += consumed;
            op += written;
        }

        // Tail: copy the remaining < 8 bytes into a zero-padded buffer.
        let mut buf = [0u8; 16];
        let rem = n - ip;
        buf[..rem].copy_from_slice(&input[ip..]);
        let mut tp = 0;
        while tp < rem {
            let word = u64::from_le_bytes(buf[tp..tp + 8].try_into().expect("8 bytes"));
            // SAFETY: same bound as above: op <= start + 2 * (ip + tp) and ip + tp < n.
            let (consumed, written) = unsafe { self.step(word, rem - tp, base.add(op)) };
            tp += consumed;
            op += written;
        }
        // SAFETY: every byte in start..op was written by `step`, and op <= capacity.
        unsafe { out.set_len(op) };
    }

    /// Encodes one symbol (or escape) at the start of `word` and writes its
    /// code(s) to `dst`. `avail` is how many bytes of `word` are real input.
    /// Returns (input bytes consumed, output bytes written).
    ///
    /// # Safety
    /// `dst` must be valid for writing 2 bytes.
    #[inline(always)]
    unsafe fn step(&self, word: u64, avail: usize, dst: *mut u8) -> (usize, usize) {
        // SAFETY: `slot` returns `prefix * mul >> shift` with shift = 64 - bits,
        // which is < 2^bits = hash.len().
        let e = unsafe { self.hash.get_unchecked(self.slot(word)) };
        if e.len != 0 && e.len as usize <= avail && word & (u64::MAX >> e.ignored_bits) == e.sym {
            // SAFETY: caller guarantees 2 writable bytes.
            unsafe { dst.write(e.code) };
            return (e.len as usize, 1);
        }
        let entry = if avail >= 2 {
            self.short[(word & 0xFFFF) as usize]
        } else {
            self.byte[(word & 0xFF) as usize]
        };
        let code = entry & 0xFF;
        let consumed = ((entry >> SHORT_LEN_SHIFT) & 0xF) as usize;
        let escaped = ((entry & SHORT_ESC_BIT) != 0) as usize;
        // Branch-free: always store the code plus the literal byte, and only
        // advance past the literal when this was an escape.
        let pair = code | (((word & 0xFF) as u16) << 8);
        // SAFETY: caller guarantees 2 writable bytes.
        unsafe { dst.cast::<u16>().write_unaligned(pair.to_le()) };
        (consumed, 1 + escaped)
    }
}

/// The first 3 bytes of a little-endian word.
#[inline(always)]
fn prefix_key(word: u64) -> u32 {
    (word & 0x00FF_FFFF) as u32
}

/// Builds a collision-free hash table for the long symbols. Because the
/// dictionary is static, this search runs once when the encoder is created.
fn build_hash(long: &[(u32, HashEntry)]) -> Result<(Box<[HashEntry]>, u64, u32)> {
    for bits in 10..=16u32 {
        if (1usize << bits) < long.len() * 2 {
            continue;
        }
        let shift = 64 - bits;
        let mut table = vec![HashEntry::default(); 1 << bits].into_boxed_slice();
        for attempt in 0..HASH_ATTEMPTS {
            let mul = hash_multiplier(attempt);
            table.fill(HashEntry::default());
            let ok = long.iter().all(|&(key, entry)| {
                let slot = (key as u64).wrapping_mul(mul).wrapping_shr(shift) as usize;
                if table[slot].len != 0 {
                    return false;
                }
                table[slot] = entry;
                true
            });
            if ok {
                return Ok((table, mul, shift));
            }
        }
    }
    Err(Error::InvalidDictionary("could not build a collision-free hash table"))
}

/// Prepared decoder tables for a U8 dictionary. Plain data, so it can be
/// copied to other devices as is.
#[repr(C, align(64))]
pub(crate) struct U8Decoder {
    /// Symbol bytes per code, little-endian, zero padded.
    sym: [u64; 256],
    /// Symbol length per code; 0 for codes that are not in the dictionary.
    len: [u8; 256],
    /// Whether every non-escape code is a symbol (255 symbols). Then no code
    /// can be invalid and the hot loop skips the check.
    full: bool,
}

impl U8Decoder {
    pub(crate) fn new(dict: &Dictionary) -> Self {
        let mut sym = [0u64; 256];
        let mut len = [0u8; 256];
        for (code, s) in dict.symbols().enumerate() {
            sym[code] = pack_le(s);
            len[code] = s.len() as u8;
        }
        Self {
            sym,
            len,
            full: dict.len() == crate::scheme::U8_MAX_SYMBOLS,
        }
    }

    /// Decodes `codes`, appending at most `limit` bytes to `out`. Returns the
    /// number of bytes appended.
    ///
    /// Never panics or writes out of bounds on malformed input: invalid codes,
    /// truncated escapes and output larger than `limit` are reported as errors.
    pub(crate) fn decode(&self, codes: &[u8], out: &mut Vec<u8>, limit: usize) -> Result<usize> {
        if self.full {
            self.decode_impl::<false>(codes, out, limit)
        } else {
            self.decode_impl::<true>(codes, out, limit)
        }
    }

    /// `CHECK` enables invalid-code tracking in the fast path. It is only
    /// disabled for full dictionaries, where every code is valid.
    #[inline(always)]
    fn decode_impl<const CHECK: bool>(&self, codes: &[u8], out: &mut Vec<u8>, limit: usize) -> Result<usize> {
        // Slack after `limit` lets every symbol be written with one 8-byte
        // store and lets the fast path run until the very end of the output.
        out.reserve(limit.checked_add(SLACK).ok_or(Error::Corrupt("output size overflow"))?);
        let start = out.len();
        // SAFETY: reserve() guarantees capacity >= start + limit + SLACK.
        let base = unsafe { out.as_mut_ptr().add(start) };
        let n = codes.len();
        let mut ip = 0;
        let mut op = 0;
        let mut invalid = 0u8;

        loop {
            // Fast path: eight codes per iteration with room for eight
            // unconditional 8-byte stores. An escape ends the batch early: the
            // codes before it are decoded, then the literal, all inline.
            while ip + 8 <= n && op <= limit {
                // SAFETY: ip + 8 <= n.
                let word = u64::from_le(unsafe { codes.as_ptr().add(ip).cast::<u64>().read_unaligned() });
                let esc = escape_mask(word);
                if esc == 0 {
                    // Common case, kept separate so the loop is fully unrolled.
                    for k in 0..8 {
                        let c = ((word >> (8 * k)) & 0xFF) as usize;
                        let l = self.len[c] as usize;
                        // SAFETY: as below, eight stores of 8 bytes starting
                        // at op <= limit end by limit + 64 <= limit + SLACK.
                        unsafe { base.add(op).cast::<u64>().write_unaligned(self.sym[c].to_le()) };
                        if CHECK {
                            invalid |= (l == 0) as u8;
                        }
                        op += l;
                    }
                    ip += 8;
                    continue;
                }
                // Number of codes before the first escape. The lowest set bit
                // of the mask is exact (see `escape_mask`).
                let plain = (esc.trailing_zeros() / 8) as usize;
                for k in 0..plain {
                    let c = ((word >> (8 * k)) & 0xFF) as usize;
                    let l = self.len[c] as usize;
                    // SAFETY: before the batch op <= limit; each code writes 8
                    // bytes at op and advances op by at most 8, so all stores
                    // end by limit + 64 <= limit + SLACK (reserved capacity).
                    unsafe { base.add(op).cast::<u64>().write_unaligned(self.sym[c].to_le()) };
                    if CHECK {
                        invalid |= (l == 0) as u8;
                    }
                    op += l;
                }
                // Escape at ip + plain; its literal follows.
                let Some(&lit) = codes.get(ip + plain + 1) else {
                    ip += plain;
                    break; // the slow path reports the truncated escape
                };
                // SAFETY: at most 7 codes (56 bytes) were written since the
                // check op <= limit, so op + 1 <= limit + 57 <= limit + SLACK.
                unsafe { base.add(op).write(lit) };
                op += 1;
                ip += plain + 2;
            }
            // The fast path may run past `limit` (into the slack) on corrupt
            // input; stop there instead of decoding further.
            if op > limit {
                return Err(Error::Corrupt("decoded data is larger than expected"));
            }
            if ip >= n {
                break;
            }
            // Slow path: one code with full checks.
            let c = codes[ip];
            if c == U8_ESCAPE {
                let lit = *codes.get(ip + 1).ok_or(Error::Truncated)?;
                if op + 1 > limit {
                    return Err(Error::Corrupt("decoded data is larger than expected"));
                }
                // SAFETY: op + 1 <= limit <= reserved capacity.
                unsafe { base.add(op).write(lit) };
                op += 1;
                ip += 2;
            } else {
                let l = self.len[c as usize] as usize;
                if l == 0 {
                    return Err(Error::InvalidCode);
                }
                if op + l > limit {
                    return Err(Error::Corrupt("decoded data is larger than expected"));
                }
                // SAFETY: op + l <= limit, so op + 8 <= limit + 8 <= reserved capacity.
                unsafe { base.add(op).cast::<u64>().write_unaligned(self.sym[c as usize].to_le()) };
                op += l;
                ip += 1;
            }
        }
        if invalid != 0 {
            return Err(Error::InvalidCode);
        }
        // SAFETY: every byte in 0..op was written (each store writes at least
        // as many bytes as it advances op) and op <= limit <= reserved capacity.
        unsafe { out.set_len(start + op) };
        Ok(op)
    }
}

/// Bytes reserved after the expected output: 8 codes of 8 bytes each.
const SLACK: usize = 64;

/// Sets bit 7 of every byte of `word` that is the escape code 0xFF.
///
/// Uses the classic "has zero byte" trick on the inverted word. Bytes above
/// the first escape may be flagged spuriously (borrow propagation), but the
/// lowest flagged byte is always the first escape, which is all the decoder
/// uses. Returns 0 if there is no escape.
#[inline(always)]
fn escape_mask(word: u64) -> u64 {
    let x = !word; // escape bytes become zero bytes
    x.wrapping_sub(0x0101_0101_0101_0101) & !x & 0x8080_8080_8080_8080
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Scheme, reference};

    /// Regression test: with only a few fixed multipliers, about 1 in 250
    /// valid dictionaries with many long symbols could not build an encoder.
    #[test]
    fn hash_table_builds_for_random_key_sets() {
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..if cfg!(miri) { 2 } else { 2000 } {
            let mut keys = alloc::collections::BTreeSet::new();
            while keys.len() < 255 {
                keys.insert((next() & 0x00FF_FFFF) as u32);
            }
            let long: Vec<(u32, HashEntry)> = keys
                .iter()
                .map(|&k| {
                    (
                        k,
                        HashEntry {
                            len: 3,
                            ..HashEntry::default()
                        },
                    )
                })
                .collect();
            let (table, mul, shift) = build_hash(&long).unwrap();
            for &(key, _) in &long {
                let slot = (u64::from(key)).wrapping_mul(mul).wrapping_shr(shift) as usize;
                assert_eq!(table[slot].len, 3);
            }
        }
    }

    #[test]
    fn escape_detection() {
        let first = |w: u64| (escape_mask(w).trailing_zeros() / 8) as usize;
        // Exhaustive over the byte patterns that matter: 0xFF, 0xFE (borrow
        // edge case), 0x00 and an ordinary byte, in every position.
        let vals = [0xFFu8, 0xFE, 0x00, 0x41];
        for pattern in 0..4u32.pow(8) {
            let mut bytes = [0u8; 8];
            let mut p = pattern;
            for b in &mut bytes {
                *b = vals[(p % 4) as usize];
                p /= 4;
            }
            let expected = bytes.iter().position(|&b| b == 0xFF).unwrap_or(8);
            assert_eq!(first(u64::from_le_bytes(bytes)), expected, "{bytes:02x?}");
        }
    }

    #[test]
    fn matches_reference_small() {
        let dict = Dictionary::from_symbols(Scheme::U8, [&b"the "[..], b"he", b"t", b"abcdefgh", b"xyz"]).unwrap();
        let enc = U8Encoder::new(&dict).unwrap();
        let dec = U8Decoder::new(&dict);
        for input in [
            &b""[..],
            b"t",
            b"th",
            b"the",
            b"the cat the hat abcdefgh xyz!",
            b"abcdefg",
            b"\xff\xff\x00",
        ] {
            let mut codes = Vec::new();
            enc.encode(input, &mut codes);
            assert_eq!(codes, reference::encode_u8(&dict, input), "input {input:?}");
            let mut out = Vec::new();
            assert_eq!(dec.decode(&codes, &mut out, input.len()).unwrap(), input.len());
            assert_eq!(out, input);
        }
    }

    #[test]
    fn decode_rejects_bad_streams() {
        let dict = Dictionary::from_symbols(Scheme::U8, [&b"ab"[..]]).unwrap();
        let dec = U8Decoder::new(&dict);
        let mut out = Vec::new();
        assert_eq!(dec.decode(&[1], &mut out, 100), Err(Error::InvalidCode));
        assert_eq!(
            dec.decode(&[0, 0, 0, 1, 0, 0, 0, 0], &mut out, 100),
            Err(Error::InvalidCode)
        );
        assert_eq!(dec.decode(&[0, 255], &mut out, 100), Err(Error::Truncated));
        assert!(dec.decode(&[0, 0, 0], &mut out, 5).is_err());
        assert_eq!(
            dec.decode(&[0; 64], &mut out, 10),
            Err(Error::Corrupt("decoded data is larger than expected"))
        );
    }
}
