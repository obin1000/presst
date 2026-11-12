//! Runs the Vortex FSST kernels (`kernels/vortex/`) on Vortex's data layout, so
//! the warp decoder written for Vortex can be tested and compared with Vortex's
//! own decoder here. Not a stable API.

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::{GpuContext, GpuError, Result};

/// An FSST array as Vortex's CUDA executor passes it to the kernels.
#[derive(Debug, Clone, Copy)]
pub struct VortexFsst<'a> {
    /// Code streams of all strings.
    pub codes: &'a [u8],
    /// `n + 1` offsets into `codes`.
    pub code_offsets: &'a [u32],
    /// Symbols, little-endian (high bytes unspecified).
    pub symbols: &'a [u64],
    /// Symbol lengths, one per symbol.
    pub symbol_lengths: &'a [u8],
    /// `n + 1` decoded offsets (prefix sums of the uncompressed lengths).
    pub output_offsets: &'a [u64],
    /// Validity bitmap (bit i of string i, after `validity_bit_offset`).
    pub validity: &'a [u8],
    /// Bit offset of string 0 in `validity`.
    pub validity_bit_offset: u64,
}

/// Which Vortex kernel to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VortexKernel {
    /// Vortex's `fsst_u32`: one thread per string.
    ThreadPerString,
    /// `fsst_warp_u32`: one warp per 32 strings.
    WarpPerGroup,
}

/// Decoded bytes, 16-byte `BinaryView`s (as `[len, prefix|inline.., ..]` u32
/// quadruples), and the kernel time in milliseconds.
pub type VortexDecoded = (Vec<u8>, Vec<[u32; 4]>, f32);

impl GpuContext {
    /// Decodes `array` with a Vortex kernel, writing views too. Uploads the
    /// inputs first; the time covers only the kernel.
    #[doc(hidden)]
    pub fn decode_vortex_fsst(&self, array: &VortexFsst<'_>, kernel: VortexKernel) -> Result<VortexDecoded> {
        let n = array.code_offsets.len().saturating_sub(1);
        if array.output_offsets.len() != n + 1 || array.symbols.len() != array.symbol_lengths.len() {
            return Err(GpuError::Unavailable("inconsistent Vortex FSST layout"));
        }
        let total = *array.output_offsets.last().unwrap_or(&0) as usize;
        let mut codes = self.alloc_uninit::<u8>(array.codes.len() + 16)?;
        if !array.codes.is_empty() {
            self.stream
                .memcpy_htod(array.codes, &mut codes.slice_mut(..array.codes.len()))?;
        }
        let code_offsets = self.stream.clone_htod(array.code_offsets)?;
        // The Vortex kernels read symbols[code] directly; give them a full table.
        let mut symbols = array.symbols.to_vec();
        symbols.resize(256, 0);
        let mut lengths = array.symbol_lengths.to_vec();
        lengths.resize(256, 0);
        let symbols_dev = self.stream.clone_htod(&symbols)?;
        let lengths_dev = self.stream.clone_htod(&lengths)?;
        let output_offsets = self.stream.clone_htod(array.output_offsets)?;
        let mut validity = array.validity.to_vec();
        validity.push(0);
        let validity_dev = self.stream.clone_htod(&validity)?;
        let mut out = self.alloc_uninit::<u8>(total + 16)?;
        let mut views = self.alloc_uninit::<u32>(4 * n.max(1))?;
        let n64 = n as u64;
        let num_symbols = array.symbols.len() as u32;
        let (func, cfg) = match kernel {
            VortexKernel::ThreadPerString => (
                &self.vortex_fsst_thread,
                // Vortex: 64 threads per block, 32 strings per thread.
                LaunchConfig {
                    grid_dim: ((n64.div_ceil(64 * 32)).max(1) as u32, 1, 1),
                    block_dim: (64, 1, 1),
                    shared_mem_bytes: 0,
                },
            ),
            VortexKernel::WarpPerGroup => (
                &self.vortex_fsst_warp,
                LaunchConfig {
                    grid_dim: ((n64.div_ceil(32 * 8)).max(1) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
            ),
        };
        let mut launch = self.stream.launch_builder(func);
        launch
            .arg(&codes)
            .arg(&code_offsets)
            .arg(&symbols_dev)
            .arg(&lengths_dev);
        if kernel == VortexKernel::WarpPerGroup {
            launch.arg(&num_symbols);
        }
        launch
            .arg(&output_offsets)
            .arg(&validity_dev)
            .arg(&array.validity_bit_offset)
            .arg(&mut out)
            .arg(&mut views)
            .arg(&n64)
            .record_kernel_launch(CUevent_flags::CU_EVENT_DEFAULT);
        // SAFETY: the arguments match `fsst_u32` / `fsst_warp_u32` (the warp
        // kernel also takes the symbol count). The symbol table is padded to 256
        // entries, the output offsets are the caller's prefix sums with `total`
        // bytes (plus 16 of slack) allocated, and `views` holds n entries.
        let events = unsafe { launch.launch(cfg) }?;
        let ms = match events {
            Some((start, end)) => start.elapsed_ms(&end)?,
            None => 0.0,
        };
        let mut bytes = self.stream.clone_dtoh(&out)?;
        bytes.truncate(total);
        let raw_views = self.stream.clone_dtoh(&views)?;
        let views = raw_views[..4 * n]
            .chunks_exact(4)
            .map(|v| [v[0], v[1], v[2], v[3]])
            .collect();
        Ok((bytes, views, ms))
    }
}
