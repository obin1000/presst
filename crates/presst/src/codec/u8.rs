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

/// Multipliers tried when searching for a collision-free hash function.
const HASH_MULTIPLIERS: [u64; 6] = [
    2_971_215_073,
    0x9E37_79B9_7F4A_7C15,
    0xC2B2_AE3D_27D4_EB4F,
    0x1656_67B1_9E37_79F9,
    0x27D4_EB2F_1656_67C5,
    0xFF51_AFD7_ED55_8CCD,
];

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
        for &mul in &HASH_MULTIPLIERS {
            let shift = 64 - bits;
            let mut table = vec![HashEntry::default(); 1 << bits].into_boxed_slice();
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
}

impl U8Decoder {
    pub(crate) fn new(dict: &Dictionary) -> Self {
        let mut sym = [0u64; 256];
        let mut len = [0u8; 256];
        for (code, s) in dict.symbols().enumerate() {
            sym[code] = pack_le(s);
            len[code] = s.len() as u8;
        }
        Self { sym, len }
    }

    /// Decodes `codes`, appending at most `limit` bytes to `out`. Returns the
    /// number of bytes appended.
    ///
    /// Never panics or writes out of bounds on malformed input: invalid codes,
    /// truncated escapes and output larger than `limit` are reported as errors.
    pub(crate) fn decode(&self, codes: &[u8], out: &mut Vec<u8>, limit: usize) -> Result<usize> {
        // 8 bytes of slack let every symbol be written with one 8-byte store.
        out.reserve(limit.checked_add(8).ok_or(Error::Corrupt("output size overflow"))?);
        let start = out.len();
        // SAFETY: reserve() guarantees capacity >= start + limit + 8.
        let base = unsafe { out.as_mut_ptr().add(start) };
        let n = codes.len();
        let mut ip = 0;
        let mut op = 0;
        let mut invalid = 0u8;

        loop {
            // Fast path: four codes per iteration, no escapes, and room for
            // four unconditional 8-byte stores.
            while ip + 4 <= n && op + 32 <= limit {
                // SAFETY: ip + 4 <= n.
                let word = u32::from_le(unsafe { codes.as_ptr().add(ip).cast::<u32>().read_unaligned() });
                if has_escape(word) {
                    break;
                }
                for k in 0..4 {
                    let c = ((word >> (8 * k)) & 0xFF) as usize;
                    let l = self.len[c] as usize;
                    // SAFETY: before the 4 codes op + 32 <= limit and each code
                    // writes 8 bytes at op and advances op by at most 8, so all
                    // stores end at or before limit + 8 <= reserved capacity.
                    unsafe { base.add(op).cast::<u64>().write_unaligned(self.sym[c].to_le()) };
                    invalid |= (l == 0) as u8;
                    op += l;
                }
                ip += 4;
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

/// Whether any byte of `word` is the escape code 0xFF.
#[inline(always)]
fn has_escape(word: u32) -> bool {
    let x = !word; // escape bytes become zero bytes
    x.wrapping_sub(0x0101_0101) & !x & 0x8080_8080 != 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Scheme, reference};

    #[test]
    fn escape_detection() {
        assert!(!has_escape(0x0001_02FE));
        assert!(has_escape(0xFF00_0000));
        assert!(has_escape(0x0000_00FF));
        assert!(has_escape(0x01FF_0101));
        assert!(!has_escape(0));
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
