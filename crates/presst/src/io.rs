//! Streaming frame reader and writer on top of [`std::io`].

use std::io::{self, Read, Write};
use std::vec::Vec;

use crate::frame::{self, BLOCK_HEADER_LEN, BlockHeader, ContentHasher, FLAG_CONTENT_SIZE, FrameHeader, FrameOptions};
use crate::{Compressor, Decompressor, Error};

/// Compresses a byte stream into a frame, one block at a time.
///
/// Call [`finish`](Self::finish) when done: it writes the last block and the
/// checksum. Dropping the writer without finishing leaves an incomplete frame.
///
/// ```
/// use std::io::{Read, Write};
/// use presst::io::{FrameReader, FrameWriter};
/// use presst::{Compressor, Decompressor, Dictionary, FrameOptions, Scheme};
///
/// let dict = Dictionary::from_symbols(Scheme::U8, [&b"log "[..], b"line"])?;
/// let mut w = FrameWriter::new(Compressor::new(&dict)?, Vec::new(), FrameOptions::default());
/// w.write_all(b"log line 1\nlog line 2\n")?;
/// let frame = w.finish()?;
///
/// let mut text = String::new();
/// FrameReader::new(Decompressor::new(&dict)?, &frame[..])?.read_to_string(&mut text)?;
/// assert_eq!(text, "log line 1\nlog line 2\n");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct FrameWriter<W: Write> {
    comp: Compressor,
    inner: W,
    block_size: usize,
    buf: Vec<u8>,
    out: Vec<u8>,
    hasher: Option<ContentHasherDebug>,
}

/// Wrapper so `FrameWriter` can derive `Debug`.
struct ContentHasherDebug(ContentHasher);

impl core::fmt::Debug for ContentHasherDebug {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("ContentHasher")
    }
}

impl<W: Write> FrameWriter<W> {
    /// Creates a writer. `options.content_size` is ignored because the size is
    /// not known up front.
    pub fn new(comp: Compressor, inner: W, options: FrameOptions) -> Self {
        let block_size = options.block_size();
        let header = FrameHeader {
            scheme: comp.scheme(),
            block_size_log2: block_size.trailing_zeros() as u8,
            dict_id: comp.dict_id(),
            checksum: options.checksum,
            content_size: None,
        };
        let mut out = Vec::new();
        header.write(&mut out);
        Self {
            comp,
            inner,
            block_size,
            buf: Vec::with_capacity(block_size),
            out,
            hasher: options.checksum.then(|| ContentHasherDebug(ContentHasher::new())),
        }
    }

    /// Writes the last block and the checksum, flushes, and returns the inner writer.
    pub fn finish(mut self) -> io::Result<W> {
        frame::write_block(&self.comp, &self.buf, true, &mut self.out);
        if let Some(h) = &self.hasher {
            self.out.extend_from_slice(&h.0.finish().to_le_bytes());
        }
        self.inner.write_all(&self.out)?;
        self.inner.flush()?;
        Ok(self.inner)
    }

    fn flush_full_blocks(&mut self) -> io::Result<()> {
        // Keep the last (possibly full) block buffered: only `finish` knows
        // which block is the last one.
        let mut start = 0;
        while self.buf.len() - start > self.block_size {
            frame::write_block(
                &self.comp,
                &self.buf[start..start + self.block_size],
                false,
                &mut self.out,
            );
            start += self.block_size;
        }
        self.buf.drain(..start);
        if !self.out.is_empty() {
            self.inner.write_all(&self.out)?;
            self.out.clear();
        }
        Ok(())
    }
}

impl<W: Write> Write for FrameWriter<W> {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        if let Some(h) = &mut self.hasher {
            h.0.update(data);
        }
        self.buf.extend_from_slice(data);
        if self.buf.len() > self.block_size {
            self.flush_full_blocks()?;
        }
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.write_all(&self.out)?;
        self.out.clear();
        self.inner.flush()
    }
}

/// Decompresses a frame from a byte stream, one block at a time.
///
/// Stops at the end of the frame; bytes after it are left in the inner reader.
pub struct FrameReader<R: Read> {
    dec: Decompressor,
    inner: R,
    header: FrameHeader,
    block: Vec<u8>,
    pos: usize,
    scratch: Vec<u8>,
    total: u64,
    hasher: Option<ContentHasher>,
    done: bool,
}

impl<R: Read> core::fmt::Debug for FrameReader<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FrameReader")
            .field("header", &self.header)
            .field("done", &self.done)
            .finish_non_exhaustive()
    }
}

impl<R: Read> FrameReader<R> {
    /// Reads and validates the frame header.
    pub fn new(dec: Decompressor, mut inner: R) -> io::Result<Self> {
        let mut head = [0u8; 24];
        inner.read_exact(&mut head[..16])?;
        let len = if head[5] & FLAG_CONTENT_SIZE != 0 {
            inner.read_exact(&mut head[16..24])?;
            24
        } else {
            16
        };
        let (header, _) = FrameHeader::parse(&head[..len])?;
        if header.scheme != dec.scheme() {
            return Err(Error::Corrupt("frame scheme does not match dictionary").into());
        }
        dec.check_dict(header.dict_id)?;
        let hasher = header.checksum.then(ContentHasher::new);
        Ok(Self {
            dec,
            inner,
            header,
            block: Vec::new(),
            pos: 0,
            scratch: Vec::new(),
            total: 0,
            hasher,
            done: false,
        })
    }

    /// The parsed frame header.
    pub fn header(&self) -> &FrameHeader {
        &self.header
    }

    /// Returns the inner reader, positioned after the bytes consumed so far.
    pub fn into_inner(self) -> R {
        self.inner
    }

    fn next_block(&mut self) -> io::Result<()> {
        let mut bh = [0u8; BLOCK_HEADER_LEN];
        self.inner.read_exact(&mut bh)?;
        let bh = BlockHeader::parse(bh, &self.header, self.dec.max_expansion())?;
        self.block.clear();
        self.pos = 0;
        if bh.raw {
            self.block.resize(bh.csize, 0);
            self.inner.read_exact(&mut self.block)?;
        } else {
            self.scratch.resize(bh.csize, 0);
            self.inner.read_exact(&mut self.scratch)?;
            self.dec
                .decompress_block_into(&self.scratch, bh.dsize, &mut self.block)?;
        }
        if let Some(h) = &mut self.hasher {
            h.update(&self.block);
        }
        self.total += self.block.len() as u64;
        if bh.last {
            self.done = true;
            if let Some(size) = self.header.content_size {
                if size != self.total {
                    return Err(Error::Corrupt("content size does not match").into());
                }
            }
            if let Some(h) = &self.hasher {
                let mut sum = [0u8; 8];
                self.inner.read_exact(&mut sum)?;
                if u64::from_le_bytes(sum) != h.finish() {
                    return Err(Error::ChecksumMismatch.into());
                }
            }
        }
        Ok(())
    }
}

impl<R: Read> Read for FrameReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        while self.pos == self.block.len() {
            if self.done || buf.is_empty() {
                return Ok(0);
            }
            self.next_block()?;
        }
        let n = buf.len().min(self.block.len() - self.pos);
        buf[..n].copy_from_slice(&self.block[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Dictionary, Scheme};

    #[test]
    fn streaming_matches_one_shot() {
        let dict = Dictionary::from_symbols(Scheme::U8, [&b"stream "[..], b"ing"]).unwrap();
        let comp = Compressor::new(&dict).unwrap();
        let dec = Decompressor::new(&dict).unwrap();
        let input: Vec<u8> = b"streaming stream of bytes "
            .iter()
            .copied()
            .cycle()
            .take(10_000)
            .collect();
        for opts in [
            FrameOptions {
                block_size_log2: 10,
                ..FrameOptions::default()
            },
            FrameOptions::default(),
        ] {
            for piece in [1usize, 7, 1024, 4096, 20_000] {
                let mut w = FrameWriter::new(comp.clone(), Vec::new(), opts);
                for chunk in input.chunks(piece) {
                    w.write_all(chunk).unwrap();
                }
                let frame = w.finish().unwrap();
                assert_eq!(dec.decompress(&frame).unwrap(), input, "one-shot decode, piece {piece}");
                let mut back = Vec::new();
                let mut trailing = frame.clone();
                trailing.extend_from_slice(b"tail");
                let mut r = FrameReader::new(dec.clone(), &trailing[..]).unwrap();
                r.read_to_end(&mut back).unwrap();
                assert_eq!(back, input);
                let mut rest = Vec::new();
                r.into_inner().read_to_end(&mut rest).unwrap();
                assert_eq!(rest, b"tail");
            }
        }
        // Exact multiple of the block size, and empty input.
        for len in [0usize, 1024, 2048] {
            let mut w = FrameWriter::new(
                comp.clone(),
                Vec::new(),
                FrameOptions {
                    block_size_log2: 10,
                    ..FrameOptions::default()
                },
            );
            w.write_all(&input[..len]).unwrap();
            let frame = w.finish().unwrap();
            assert_eq!(dec.decompress(&frame).unwrap(), &input[..len]);
        }
    }

    #[test]
    fn reader_detects_corruption() {
        let dict = Dictionary::from_symbols(Scheme::U8, [&b"abc"[..]]).unwrap();
        let frame = Compressor::new(&dict).unwrap().compress(b"abcabcabc");
        let dec = Decompressor::new(&dict).unwrap();
        let mut bad = frame.clone();
        let n = bad.len();
        bad[n - 1] ^= 1;
        let mut out = Vec::new();
        assert!(
            FrameReader::new(dec.clone(), &bad[..])
                .unwrap()
                .read_to_end(&mut out)
                .is_err()
        );
        assert!(
            FrameReader::new(dec, &frame[..n - 3])
                .unwrap()
                .read_to_end(&mut out)
                .is_err()
        );
    }
}
