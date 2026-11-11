//! `.vals` files: a column of byte strings, each prefixed by its length as a
//! little-endian `u32`. Values may contain newlines or any other bytes.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Writes `values` to `path`.
pub fn write_vals<I, S>(path: &Path, values: I) -> io::Result<()>
where
    I: IntoIterator<Item = S>,
    S: AsRef<[u8]>,
{
    let mut w = BufWriter::new(File::create(path)?);
    for v in values {
        let v = v.as_ref();
        let len = u32::try_from(v.len()).map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "value too long"))?;
        w.write_all(&len.to_le_bytes())?;
        w.write_all(v)?;
    }
    w.flush()
}

/// Reads all values from `path`.
pub fn read_vals(path: &Path) -> io::Result<Vec<Vec<u8>>> {
    let mut data = Vec::new();
    BufReader::new(File::open(path)?).read_to_end(&mut data)?;
    parse_vals(&data)
}

/// Parses the `.vals` encoding.
pub fn parse_vals(mut data: &[u8]) -> io::Result<Vec<Vec<u8>>> {
    let bad = || io::Error::new(io::ErrorKind::InvalidData, "truncated .vals file");
    let mut out = Vec::new();
    while !data.is_empty() {
        let (len, rest) = data.split_first_chunk::<4>().ok_or_else(bad)?;
        let len = u32::from_le_bytes(*len) as usize;
        if rest.len() < len {
            return Err(bad());
        }
        out.push(rest[..len].to_vec());
        data = &rest[len..];
    }
    Ok(out)
}

/// Writes one `i64` per value (for example a timestamp), little-endian.
pub fn write_i64s(path: &Path, values: &[i64]) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    for v in values {
        w.write_all(&v.to_le_bytes())?;
    }
    w.flush()
}

/// Reads a file written by [`write_i64s`].
pub fn read_i64s(path: &Path) -> io::Result<Vec<i64>> {
    let data = std::fs::read(path)?;
    if data.len() % 8 != 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "truncated i64 file"));
    }
    Ok(data
        .chunks_exact(8)
        .map(|c| {
            let mut b = [0u8; 8];
            b.copy_from_slice(c);
            i64::from_le_bytes(b)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let dir = std::env::temp_dir().join(format!("presst-vals-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let values: Vec<Vec<u8>> = vec![b"".to_vec(), b"a\nb".to_vec(), vec![0, 255, 10], b"hello".to_vec()];
        let p = dir.join("x.vals");
        write_vals(&p, &values).unwrap();
        assert_eq!(read_vals(&p).unwrap(), values);
        let t = dir.join("x.time");
        write_i64s(&t, &[1, -5, i64::MIN]).unwrap();
        assert_eq!(read_i64s(&t).unwrap(), [1, -5, i64::MIN]);
        assert!(parse_vals(&[5, 0, 0, 0, 1]).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
