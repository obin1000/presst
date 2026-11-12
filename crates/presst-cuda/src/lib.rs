//! CUDA backend for presst: decode (and later encode) on NVIDIA GPUs.
//!
//! The kernels read the same formats as the CPU codec and produce byte-identical
//! output. They are fastest on **format version 2** data: frames written with
//! [`FrameOptions::split_size_log2`](presst::FrameOptions) and string batches
//! with a decoded index, because every split or group then decodes
//! independently at a known output position.
//!
//! ```no_run
//! use presst::{Compressor, Dictionary, FrameOptions, Scheme};
//! use presst_cuda::{DecodeKernel, GpuContext};
//!
//! let dict = Dictionary::from_symbols(Scheme::U8, [&b"lorem "[..], b"ipsum "])?;
//! let frame = Compressor::new(&dict)?.compress_with(
//!     b"lorem ipsum lorem ipsum",
//!     FrameOptions { split_size_log2: Some(12), ..FrameOptions::default() },
//! );
//! let gpu = GpuContext::new(0)?;
//! let table = gpu.upload(&dict)?;
//! assert_eq!(gpu.decompress_frame(&table, &frame, DecodeKernel::default())?, b"lorem ipsum lorem ipsum");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! The crate always builds: without nvcc at build time, or without a GPU and
//! driver at run time, [`is_available`] returns false and [`GpuContext::new`]
//! fails with [`GpuError::Unavailable`].

mod split;

use std::fmt;
use std::sync::Arc;

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use presst::{DictId, Dictionary, FrameIndex, STRING_GROUP_SIZE_LOG2, Scheme, StringBatch};
use xxhash_rust::xxh3::xxh3_64;

pub use split::{GpuSplit, frame_splits};

/// The compiled kernels (empty if nvcc was not available at build time).
static FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/presst.fatbin"));

/// Threads per block of the decode kernels (their `__launch_bounds__`).
const DECODE_THREADS: u32 = 256;

/// Errors of the GPU backend.
#[derive(Debug)]
#[non_exhaustive]
pub enum GpuError {
    /// No kernels (built without nvcc), no CUDA driver, or no GPU.
    Unavailable(&'static str),
    /// A CUDA driver call failed.
    Cuda(DriverError),
    /// The input is not valid presst data (checked on the host).
    Format(presst::Error),
    /// The kernel found invalid code streams; the value holds the error bits
    /// (see `kernels/presst_decode.cuh`).
    Corrupt(u32),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(why) => write!(f, "GPU backend unavailable: {why}"),
            Self::Cuda(e) => write!(f, "CUDA error: {e}"),
            Self::Format(e) => write!(f, "{e}"),
            Self::Corrupt(bits) => write!(f, "corrupt code stream (GPU error bits {bits:#x})"),
        }
    }
}

impl std::error::Error for GpuError {}

impl From<DriverError> for GpuError {
    fn from(e: DriverError) -> Self {
        Self::Cuda(e)
    }
}

impl From<presst::Error> for GpuError {
    fn from(e: presst::Error) -> Self {
        Self::Format(e)
    }
}

/// Decoded strings: the concatenated bytes and `len + 1` offsets into them.
pub type DecodedStrings = (Vec<u8>, Vec<u32>);

/// Result type of the GPU backend.
pub type Result<T, E = GpuError> = std::result::Result<T, E>;

/// Whether the kernels were compiled into this build.
#[must_use]
pub fn kernels_built() -> bool {
    cfg!(presst_cuda_kernels) && !FATBIN.is_empty()
}

/// Whether the GPU backend can run here: kernels built, CUDA driver present,
/// and at least one GPU.
#[must_use]
pub fn is_available() -> bool {
    // SAFETY: only probes whether the driver library can be loaded.
    kernels_built()
        && unsafe { cudarc::driver::sys::is_culib_present() }
        && CudaContext::device_count().is_ok_and(|n| n > 0)
}

/// Which decode kernel to run, and where it reads the 2.3 KB symbol table
/// from (shared memory, copied per thread block, or global memory through
/// the read-only L1 cache).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecodeKernel {
    /// K1: one thread per split, table in shared memory.
    ThreadShared,
    /// K1: one thread per split, table through L1.
    ThreadGlobal,
    /// K2: one warp per split (coalesced code loads, warp scans, staged
    /// 16-byte stores), table in shared memory.
    WarpShared,
    /// K2 with the table through L1.
    #[default]
    WarpGlobal,
}

impl DecodeKernel {
    /// Every kernel variant.
    pub const ALL: [Self; 4] = [
        Self::ThreadShared,
        Self::ThreadGlobal,
        Self::WarpShared,
        Self::WarpGlobal,
    ];

    fn symbol(self) -> &'static str {
        match self {
            Self::ThreadShared => "presst_decode_splits_shared",
            Self::ThreadGlobal => "presst_decode_splits_global",
            Self::WarpShared => "presst_decode_warp_shared",
            Self::WarpGlobal => "presst_decode_warp_global",
        }
    }

    /// Splits handled by one thread block of 256 threads.
    fn splits_per_block(self) -> u32 {
        match self {
            Self::ThreadShared | Self::ThreadGlobal => DECODE_THREADS,
            Self::WarpShared | Self::WarpGlobal => DECODE_THREADS / 32,
        }
    }
}

/// A GPU, a stream on it and the loaded kernels.
pub struct GpuContext {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    /// Indexed like [`DecodeKernel::ALL`].
    decode: Vec<CudaFunction>,
    /// String batch kernels: table in shared memory, through L1.
    strings_shared: CudaFunction,
    strings_global: CudaFunction,
}

impl fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuContext")
            .field("ordinal", &self.ctx.ordinal())
            .finish_non_exhaustive()
    }
}

/// A dictionary's decode table, resident on the GPU.
#[derive(Debug)]
pub struct GpuDictionary {
    id: DictId,
    table: CudaSlice<u8>,
    /// Symbol lengths, for computing decoded sizes on the host.
    lens: [u8; 256],
}

impl GpuDictionary {
    /// The dictionary's id.
    #[must_use]
    pub fn id(&self) -> DictId {
        self.id
    }
}

/// The `DecodeTable` of `kernels/presst_decode.cuh`: 256 little-endian u64
/// symbols (zero padded) followed by 256 lengths (0 for unused codes).
fn decode_table_bytes(dict: &Dictionary) -> Vec<u8> {
    let mut sym = [0u64; 256];
    let mut len = [0u8; 256];
    for (code, s) in dict.symbols().enumerate() {
        let mut word = [0u8; 8];
        word[..s.len()].copy_from_slice(s);
        sym[code] = u64::from_le_bytes(word);
        len[code] = s.len() as u8;
    }
    let mut out = Vec::with_capacity(256 * 9);
    for s in sym {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out.extend_from_slice(&len);
    out
}

/// Timing of one GPU operation.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GpuTiming {
    /// Kernel time in milliseconds (CUDA events), excluding transfers.
    pub kernel_ms: f32,
}

impl GpuContext {
    /// Opens GPU `ordinal` and loads the kernels.
    pub fn new(ordinal: usize) -> Result<Self> {
        if !kernels_built() {
            return Err(GpuError::Unavailable("presst-cuda was built without nvcc"));
        }
        // SAFETY: only probes whether the driver library can be loaded.
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            return Err(GpuError::Unavailable("no CUDA driver"));
        }
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();
        let module = ctx.load_module(Ptx::from_binary(FATBIN.to_vec()))?;
        Ok(Self {
            decode: DecodeKernel::ALL
                .iter()
                .map(|k| module.load_function(k.symbol()))
                .collect::<std::result::Result<_, _>>()?,
            strings_shared: module.load_function("presst_decode_strings_shared")?,
            strings_global: module.load_function("presst_decode_strings_global")?,
            _module: module,
            stream,
            ctx,
        })
    }

    /// The GPU's name.
    pub fn device_name(&self) -> Result<String> {
        Ok(self.ctx.name()?)
    }

    /// Uploads a dictionary's decode table.
    pub fn upload(&self, dict: &Dictionary) -> Result<GpuDictionary> {
        if dict.scheme() != Scheme::U8 {
            return Err(GpuError::Unavailable("only the U8 scheme is supported on the GPU"));
        }
        let mut lens = [0u8; 256];
        for (l, &n) in lens.iter_mut().zip(dict.symbol_lens()) {
            *l = n;
        }
        Ok(GpuDictionary {
            lens,
            id: dict.id(),
            table: self.stream.clone_htod(&decode_table_bytes(dict))?,
        })
    }

    /// Decompresses a frame held in host memory and returns the content.
    ///
    /// The frame is validated on the host first ([`FrameIndex::parse`]); the
    /// kernel checks the code streams. A stored checksum is verified.
    pub fn decompress_frame(&self, dict: &GpuDictionary, frame: &[u8], kernel: DecodeKernel) -> Result<Vec<u8>> {
        Ok(self.decompress_frame_timed(dict, frame, kernel)?.0)
    }

    /// Like [`decompress_frame`](Self::decompress_frame), also returning the
    /// kernel time.
    pub fn decompress_frame_timed(
        &self,
        dict: &GpuDictionary,
        frame: &[u8],
        kernel: DecodeKernel,
    ) -> Result<(Vec<u8>, GpuTiming)> {
        let index = FrameIndex::parse(frame)?;
        if index.header.dict_id != dict.id {
            return Err(GpuError::Format(presst::Error::DictionaryMismatch {
                expected: index.header.dict_id,
                actual: dict.id,
            }));
        }
        let splits = frame_splits(&index);
        let input = self.stream.clone_htod(frame)?;
        let splits_dev = self.stream.clone_htod(&split::to_bytes(&splits))?;
        // 8 bytes of slack: the kernel writes whole aligned words.
        let mut out = self.stream.alloc_zeros::<u8>(index.content_size + 8)?;
        let timing = self.launch_decode(dict, &input, &splits_dev, splits.len(), &mut out, kernel)?;
        let mut content = self.stream.clone_dtoh(&out)?;
        content.truncate(index.content_size);
        if let Some(expected) = index.checksum {
            if expected != xxh3_64(&content) {
                return Err(GpuError::Format(presst::Error::ChecksumMismatch));
            }
        }
        Ok((content, timing))
    }

    /// Runs the decode kernel over device-resident input and splits, writing
    /// into `out` (which needs 8 bytes of slack). Returns the kernel time.
    pub fn launch_decode(
        &self,
        dict: &GpuDictionary,
        input: &CudaSlice<u8>,
        splits: &CudaSlice<u8>,
        n_splits: usize,
        out: &mut CudaSlice<u8>,
        kernel: DecodeKernel,
    ) -> Result<GpuTiming> {
        let mut error = self.stream.alloc_zeros::<u32>(1)?;
        if n_splits == 0 {
            return Ok(GpuTiming::default());
        }
        let n = u32::try_from(n_splits).map_err(|_| GpuError::Unavailable("too many splits for one launch"))?;
        let func = &self.decode[DecodeKernel::ALL.iter().position(|k| *k == kernel).unwrap_or(0)];
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(kernel.splits_per_block()), 1, 1),
            block_dim: (DECODE_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(func);
        launch
            .arg(input)
            .arg(splits)
            .arg(&n)
            .arg(&dict.table)
            .arg(out)
            .arg(&mut error)
            .record_kernel_launch(CUevent_flags::CU_EVENT_DEFAULT);
        // SAFETY: the kernel signature matches these arguments (pointers to the
        // input bytes, the split array, its length, the decode table, the output
        // and the error word). Splits come from a validated `FrameIndex`: their
        // input ranges lie inside `input`, their output ranges inside `out`
        // (content size + 8 bytes of slack), and the kernel never writes past a
        // split's 8-byte-rounded output range.
        let events = unsafe { launch.launch(cfg) }?;
        let kernel_ms = match events {
            Some((start, end)) => start.elapsed_ms(&end)?,
            None => 0.0,
        };
        let error = self.stream.clone_dtoh(&error)?[0];
        if error != 0 {
            return Err(GpuError::Corrupt(error));
        }
        Ok(GpuTiming { kernel_ms })
    }

    /// Decompresses every string of a batch held in host memory. Returns the
    /// concatenated strings and `len + 1` offsets into them (Arrow layout), like
    /// [`presst::Decompressor::decompress_strings`].
    ///
    /// One warp decodes each group of 32 strings. Batches with a decoded index
    /// (format version 2, [`presst::StringOptions::decoded_index`]) are used as
    /// they are; for other batches the group starts are computed on the host
    /// first. Only the table placement of `kernel` matters here.
    pub fn decompress_strings(
        &self,
        dict: &GpuDictionary,
        batch: &StringBatch,
        kernel: DecodeKernel,
    ) -> Result<DecodedStrings> {
        Ok(self.decompress_strings_timed(dict, batch, kernel)?.0)
    }

    /// Like [`decompress_strings`](Self::decompress_strings), also returning
    /// the kernel time.
    pub fn decompress_strings_timed(
        &self,
        dict: &GpuDictionary,
        batch: &StringBatch,
        kernel: DecodeKernel,
    ) -> Result<(DecodedStrings, GpuTiming)> {
        if batch.dict_id() != dict.id {
            return Err(GpuError::Format(presst::Error::DictionaryMismatch {
                expected: batch.dict_id(),
                actual: dict.id,
            }));
        }
        let n = batch.len();
        let starts = match batch.decoded_index() {
            Some((g, starts)) if g == STRING_GROUP_SIZE_LOG2 => starts.to_vec(),
            _ => split::group_starts(batch, &dict.lens)?,
        };
        let total = *starts.last().unwrap_or(&0);
        let total = u32::try_from(total).map_err(|_| GpuError::Unavailable("decoded batch larger than 4 GiB"))?;
        let groups = split::string_groups(batch, &starts);
        if groups.is_empty() {
            return Ok(((Vec::new(), vec![0; n + 1]), GpuTiming::default()));
        }
        let codes = self.stream.clone_htod(batch.code_bytes())?;
        let code_offsets = self.stream.clone_htod(batch.offsets())?;
        let groups_dev = self.stream.clone_htod(&split::to_bytes(&groups))?;
        let mut out = self.stream.alloc_zeros::<u8>(total as usize + 16)?;
        let mut out_offsets = self.stream.alloc_zeros::<u32>(n + 1)?;
        let mut error = self.stream.alloc_zeros::<u32>(1)?;
        let n_groups = groups.len() as u32;
        let n_strings = n as u32;
        let func = match kernel {
            DecodeKernel::ThreadShared | DecodeKernel::WarpShared => &self.strings_shared,
            DecodeKernel::ThreadGlobal | DecodeKernel::WarpGlobal => &self.strings_global,
        };
        let cfg = LaunchConfig {
            grid_dim: (n_groups.div_ceil(DECODE_THREADS / 32), 1, 1),
            block_dim: (DECODE_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(func);
        launch
            .arg(&codes)
            .arg(&groups_dev)
            .arg(&n_groups)
            .arg(&dict.table)
            .arg(&mut out)
            .arg(&mut error)
            .arg(&code_offsets)
            .arg(&n_strings)
            .arg(&mut out_offsets)
            .record_kernel_launch(CUevent_flags::CU_EVENT_DEFAULT);
        // SAFETY: the arguments match the kernel signature. Groups come from the
        // batch's validated code offsets and the (validated or host-computed)
        // group starts: their code ranges lie inside `codes`, their output ranges
        // inside `out` (total + 16 bytes), and the kernel bounds every group by its
        // decoded size. `out_offsets` has n + 1 entries; group i writes entries
        // 32i .. 32i + 31 of at most n.
        let events = unsafe { launch.launch(cfg) }?;
        let kernel_ms = match events {
            Some((start, end)) => start.elapsed_ms(&end)?,
            None => 0.0,
        };
        let error = self.stream.clone_dtoh(&error)?[0];
        if error != 0 {
            return Err(GpuError::Corrupt(error));
        }
        let mut data = self.stream.clone_dtoh(&out)?;
        data.truncate(total as usize);
        let mut offsets = self.stream.clone_dtoh(&out_offsets)?;
        offsets[n] = total;
        Ok(((data, offsets), GpuTiming { kernel_ms }))
    }

    /// The stream all work is queued on.
    #[must_use]
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_table_layout() {
        let dict = Dictionary::from_symbols(Scheme::U8, [&b"ab"[..], b"c"]).unwrap();
        let t = decode_table_bytes(&dict);
        assert_eq!(t.len(), 2304);
        assert_eq!(&t[..8], b"ab\0\0\0\0\0\0");
        assert_eq!(&t[8..16], b"c\0\0\0\0\0\0\0");
        assert_eq!(&t[2048..2051], &[2, 1, 0]);
    }
}
