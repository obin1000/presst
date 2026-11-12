use core::fmt;

use crate::DictId;

/// Result type used throughout presst.
pub type Result<T, E = Error> = core::result::Result<T, E>;

/// Errors returned when building dictionaries or decoding data.
///
/// Decoding never panics on malformed input; it returns one of these instead.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The input ended before a complete structure could be read.
    Truncated,
    /// The input does not start with the expected magic bytes.
    BadMagic,
    /// The format version is newer than this library supports.
    UnsupportedVersion(u16),
    /// The code scheme is unknown or not implemented yet.
    UnsupportedScheme(u8),
    /// Frame flags that this version does not support are set.
    UnsupportedFlags(u8),
    /// The dictionary violates the constraints of its scheme.
    InvalidDictionary(&'static str),
    /// A stored checksum does not match the data.
    ChecksumMismatch,
    /// The code stream contains a code that is not in the dictionary.
    InvalidCode,
    /// The frame was compressed with a different dictionary.
    DictionaryMismatch {
        /// Dictionary the frame refers to.
        expected: DictId,
        /// Dictionary that was used to decode.
        actual: DictId,
    },
    /// The registry has no dictionary with this id.
    UnknownDictionary(DictId),
    /// The data is structurally invalid.
    Corrupt(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Truncated => f.write_str("input is truncated"),
            Error::BadMagic => f.write_str("bad magic bytes"),
            Error::UnsupportedVersion(v) => write!(f, "unsupported format version {v}"),
            Error::UnsupportedScheme(s) => write!(f, "unsupported code scheme {s}"),
            Error::UnsupportedFlags(fl) => write!(f, "unsupported frame flags {fl:#04x}"),
            Error::InvalidDictionary(why) => write!(f, "invalid dictionary: {why}"),
            Error::ChecksumMismatch => f.write_str("checksum mismatch"),
            Error::InvalidCode => f.write_str("code stream contains an invalid code"),
            Error::DictionaryMismatch { expected, actual } => {
                write!(f, "frame needs dictionary {expected}, got {actual}")
            }
            Error::UnknownDictionary(id) => write!(f, "unknown dictionary {id}"),
            Error::Corrupt(why) => write!(f, "corrupt data: {why}"),
        }
    }
}

impl core::error::Error for Error {}

#[cfg(feature = "std")]
impl From<Error> for std::io::Error {
    fn from(e: Error) -> Self {
        std::io::Error::new(std::io::ErrorKind::InvalidData, e)
    }
}
