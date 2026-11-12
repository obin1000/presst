//! # presst: pretrained static symbol table compression
//!
//! presst is an FSST-style string compressor whose symbol table is trained
//! **offline** on representative data instead of per block. A compressed
//! frame only references the dictionary by its content hash ([`DictId`]), so
//! compression needs no training step and decompression needs no table
//! parsing: the encode and decode tables are built once and stay hot.
//!
//! ```
//! use presst::{Compressor, Decompressor, Dictionary, Scheme};
//!
//! // Normally a dictionary is trained with `presst-train` and loaded from a
//! // `.pdict` file; here we build a tiny one by hand.
//! let dict = Dictionary::from_symbols(Scheme::U8, [&b"the "[..], b"quick", b"fox", b"o", b"e"])?;
//!
//! let input = b"the quick brown fox jumps over the lazy dog";
//! let frame = Compressor::new(&dict)?.compress(input);
//! let output = Decompressor::new(&dict)?.decompress(&frame)?;
//! assert_eq!(output, input);
//! # Ok::<(), presst::Error>(())
//! ```
//!
//! ## Layers
//!
//! * [`Dictionary`]: the symbols plus metadata, serialized as a `.pdict` file.
//! * [`Compressor`] / [`Decompressor`]: prepared encode/decode tables for one
//!   dictionary. Cheap to clone and share between threads.
//! * Frames ([`Compressor::compress`]): self-describing byte streams made of
//!   independently decodable blocks, see `docs/FORMAT.md`.
//! * Raw code streams ([`Compressor::compress_block_into`]) and string
//!   batches ([`Compressor::compress_strings`]) for embedding presst in
//!   columnar formats without frame overhead.
//! * [`Registry`]: looks dictionaries up by [`DictId`] when decoding frames.
//!
//! ## Code schemes
//!
//! The wire format is designed for several code widths ([`Scheme`]). Only
//! [`Scheme::U8`], which is compatible with FSST's code stream, is
//! implemented in this version.

#![no_std]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![warn(missing_docs)]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

mod codec;
mod dict;
mod error;
mod frame;
#[cfg(feature = "std")]
pub mod io;
pub mod reference;
mod registry;
mod scheme;
mod strings;
mod util;

pub use codec::{Compressor, Decompressor};
pub use dict::{DictId, Dictionary, DictionaryBuilder, Metadata, PushError};
pub use error::{Error, Result};
pub use frame::{
    BlockInfo, DEFAULT_BLOCK_SIZE_LOG2, DEFAULT_SPLIT_SIZE_LOG2, FrameHeader, FrameIndex, FrameOptions,
    MAX_BLOCK_SIZE_LOG2, MIN_BLOCK_SIZE_LOG2, MIN_SPLIT_SIZE_LOG2,
};
pub use registry::Registry;
pub use scheme::Scheme;
pub use strings::{STRING_GROUP_SIZE_LOG2, StringBatch, StringOptions};
