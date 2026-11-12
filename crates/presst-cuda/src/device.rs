//! Device-resident API: compressed and decompressed data stay in GPU memory.
//!
//! Only small metadata crosses `PCIe`: the frame layout (parsed once on the host
//! when a frame is uploaded), a 4-byte error word per kernel, and for frame
//! encoding the code length of every split.

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use presst::{FrameIndex, FrameOptions, StringBatch, block_header};

use crate::{DECODE_THREADS, DecodeKernel, GpuContext, GpuDictionary, GpuError, GpuTiming, Result, encode, split};

/// Slack every [`DeviceBytes`] buffer has after its data: the kernels read
/// 8-byte windows and whole tiles past the end, and write whole 16-byte chunks.
/// Covers the tiled encoder's look-ahead for splits of up to 64 KiB.
pub const DEVICE_SLACK: usize = (1 << 16) + 128;

/// Bytes in GPU memory, followed by [`DEVICE_SLACK`] bytes of slack.
#[derive(Debug)]
pub struct DeviceBytes {
    buf: CudaSlice<u8>,
    len: usize,
}

impl DeviceBytes {
    /// Number of valid bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether there are no valid bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The buffer (valid bytes first, then the slack).
    #[must_use]
    pub fn buffer(&self) -> &CudaSlice<u8> {
        &self.buf
    }
}

/// A frame in GPU memory, with its layout parsed and validated on the host.
#[derive(Debug)]
pub struct DeviceFrame {
    index: FrameIndex,
    bytes: CudaSlice<u8>,
    splits: CudaSlice<u8>,
    n_splits: usize,
}

impl DeviceFrame {
    /// The frame's validated index (header, blocks, content size, checksum).
    #[must_use]
    pub fn index(&self) -> &FrameIndex {
        &self.index
    }
}

/// A string batch in GPU memory, ready to decode.
#[derive(Debug)]
pub struct DeviceStringBatch {
    dict_id: presst::DictId,
    codes: CudaSlice<u8>,
    /// n + 1 code offsets.
    offsets: CudaSlice<u32>,
    groups: CudaSlice<u8>,
    n_groups: usize,
    len: usize,
    decoded_len: u32,
}

impl DeviceStringBatch {
    /// Number of strings.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the batch holds no strings.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total decoded size in bytes.
    #[must_use]
    pub fn decoded_len(&self) -> usize {
        self.decoded_len as usize
    }
}

/// Strings in GPU memory, Arrow layout: bytes plus `len + 1` u32 offsets.
#[derive(Debug)]
pub struct DeviceStrings {
    /// The concatenated strings.
    pub data: DeviceBytes,
    /// `len + 1` offsets into `data`.
    pub offsets: CudaSlice<u32>,
    /// Number of strings.
    pub len: usize,
}

/// Compressed strings in GPU memory: codes plus `len + 1` u32 offsets, the
/// layout of a [`StringBatch`].
#[derive(Debug)]
pub struct DeviceCodes {
    /// The code streams of all strings, back to back.
    pub codes: CudaSlice<u8>,
    /// Total length of the code streams.
    pub codes_len: usize,
    /// `len + 1` offsets into `codes`.
    pub offsets: CudaSlice<u32>,
    /// Number of strings.
    pub len: usize,
}

/// One thread per item, blocks of `threads`.
fn launch_1d(n: u32, threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(threads).max(1), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// One warp per item: blocks of 256 threads, 8 items each.
fn launch_warps(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(DECODE_THREADS / 32).max(1), 1, 1),
        block_dim: (DECODE_THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn events_ms(events: Option<(cudarc::driver::CudaEvent, cudarc::driver::CudaEvent)>) -> Result<f32> {
    Ok(match events {
        Some((start, end)) => start.elapsed_ms(&end)?,
        None => 0.0,
    })
}

impl GpuContext {
    /// Copies bytes to the GPU (with [`DEVICE_SLACK`]).
    pub fn upload_bytes(&self, data: &[u8]) -> Result<DeviceBytes> {
        let mut buf = self.stream.alloc_zeros::<u8>(data.len() + DEVICE_SLACK)?;
        if !data.is_empty() {
            self.stream.memcpy_htod(data, &mut buf.slice_mut(..data.len()))?;
        }
        Ok(DeviceBytes { buf, len: data.len() })
    }

    /// Allocates `len` zeroed bytes on the GPU (with [`DEVICE_SLACK`]).
    pub fn alloc_bytes(&self, len: usize) -> Result<DeviceBytes> {
        Ok(DeviceBytes {
            buf: self.stream.alloc_zeros::<u8>(len + DEVICE_SLACK)?,
            len,
        })
    }

    /// Like [`alloc_bytes`](Self::alloc_bytes), but not zeroed: for outputs
    /// that a kernel writes completely.
    fn alloc_bytes_uninit(&self, len: usize) -> Result<DeviceBytes> {
        Ok(DeviceBytes {
            buf: self.alloc_uninit::<u8>(len + DEVICE_SLACK)?,
            len,
        })
    }

    /// Copies device bytes back to the host.
    pub fn download_bytes(&self, bytes: &DeviceBytes) -> Result<Vec<u8>> {
        let mut out = vec![0u8; bytes.len];
        if bytes.len > 0 {
            self.stream.memcpy_dtoh(&bytes.buf.slice(..bytes.len), &mut out)?;
        }
        Ok(out)
    }

    /// Copies a frame to the GPU after parsing and validating it on the host.
    pub fn upload_frame(&self, frame: &[u8]) -> Result<DeviceFrame> {
        let index = FrameIndex::parse(frame)?;
        let splits = split::frame_splits(&index);
        Ok(DeviceFrame {
            bytes: self.stream.clone_htod(frame)?,
            splits: self.stream.clone_htod(&split::to_bytes(&splits))?,
            n_splits: splits.len(),
            index,
        })
    }

    /// Decodes a device frame into a new device buffer. The kernel validates
    /// the code streams; the content checksum is not verified on the GPU (it is
    /// in [`DeviceFrame::index`] for callers that want to check it).
    pub fn decode_frame_device(
        &self,
        dict: &GpuDictionary,
        frame: &DeviceFrame,
        kernel: DecodeKernel,
    ) -> Result<(DeviceBytes, GpuTiming)> {
        if frame.index.header.dict_id != dict.id {
            return Err(GpuError::Format(presst::Error::DictionaryMismatch {
                expected: frame.index.header.dict_id,
                actual: dict.id,
            }));
        }
        let mut out = self.alloc_bytes_uninit(frame.index.content_size)?;
        let timing = self.launch_decode(dict, &frame.bytes, &frame.splits, frame.n_splits, &mut out.buf, kernel)?;
        Ok((out, timing))
    }

    /// Copies a string batch to the GPU. Uses the batch's decoded index, or
    /// computes the group starts on the host if it has none.
    pub fn upload_strings(&self, dict: &GpuDictionary, batch: &StringBatch) -> Result<DeviceStringBatch> {
        if batch.dict_id() != dict.id {
            return Err(GpuError::Format(presst::Error::DictionaryMismatch {
                expected: batch.dict_id(),
                actual: dict.id,
            }));
        }
        let starts = match batch.decoded_index() {
            Some((g, starts)) if g == presst::STRING_GROUP_SIZE_LOG2 => starts.to_vec(),
            _ => split::group_starts(batch, &dict.lens)?,
        };
        let total = *starts.last().unwrap_or(&0);
        let decoded_len = u32::try_from(total).map_err(|_| GpuError::Unavailable("decoded batch larger than 4 GiB"))?;
        let groups = split::string_groups(batch, &starts);
        let mut codes = self.stream.alloc_zeros::<u8>(batch.codes_len() + 16)?;
        if batch.codes_len() > 0 {
            self.stream
                .memcpy_htod(batch.code_bytes(), &mut codes.slice_mut(..batch.codes_len()))?;
        }
        Ok(DeviceStringBatch {
            dict_id: batch.dict_id(),
            codes,
            offsets: self.stream.clone_htod(batch.offsets())?,
            groups: self.stream.clone_htod(&split::to_bytes(&groups))?,
            n_groups: groups.len(),
            len: batch.len(),
            decoded_len,
        })
    }

    /// Decodes a device string batch into device strings (Arrow layout). Only
    /// the table placement of `kernel` matters.
    pub fn decode_strings_device(
        &self,
        dict: &GpuDictionary,
        batch: &DeviceStringBatch,
        kernel: DecodeKernel,
    ) -> Result<(DeviceStrings, GpuTiming)> {
        if batch.dict_id != dict.id {
            return Err(GpuError::Format(presst::Error::DictionaryMismatch {
                expected: batch.dict_id,
                actual: dict.id,
            }));
        }
        let mut out = self.alloc_bytes_uninit(batch.decoded_len as usize)?;
        let mut offsets = self.alloc_uninit::<u32>(batch.len + 1)?;
        // The kernel writes every offset but the last; that one is the total.
        self.stream
            .memcpy_htod(&[batch.decoded_len], &mut offsets.slice_mut(batch.len..))?;
        let mut kernel_ms = 0.0;
        if batch.n_groups > 0 {
            let mut error = self.stream.alloc_zeros::<u32>(1)?;
            let n_groups = batch.n_groups as u32;
            let n_strings = batch.len as u32;
            let func = match kernel {
                DecodeKernel::ThreadShared | DecodeKernel::WarpShared => &self.strings_shared,
                DecodeKernel::ThreadGlobal | DecodeKernel::WarpGlobal => &self.strings_global,
            };
            let mut launch = self.stream.launch_builder(func);
            launch
                .arg(&batch.codes)
                .arg(&batch.groups)
                .arg(&n_groups)
                .arg(&dict.table)
                .arg(&mut out.buf)
                .arg(&mut error)
                .arg(&batch.offsets)
                .arg(&n_strings)
                .arg(&mut offsets);
            // SAFETY: the arguments match `presst_decode_strings_*`. Groups come
            // from the batch's validated code offsets and validated (or
            // host-computed) group starts, so their code ranges lie inside `codes`
            // and their output ranges inside `out` (decoded size plus slack); the
            // kernel bounds every group by its decoded size. Group i writes offsets
            // 32i .. 32i + 31 of at most n, inside `offsets` (n + 1 entries).
            let events = unsafe {
                launch
                    .record_kernel_launch(CUevent_flags::CU_EVENT_DEFAULT)
                    .launch(launch_warps(n_groups))
            }?;
            kernel_ms = events_ms(events)?;
            let error = self.stream.clone_dtoh(&error)?[0];
            if error != 0 {
                return Err(GpuError::Corrupt(error));
            }
        }
        Ok((
            DeviceStrings {
                data: out,
                offsets,
                len: batch.len,
            },
            GpuTiming { kernel_ms },
        ))
    }

    /// Exclusive prefix sum of `n` u32 values on the device: `n + 1` u64 values
    /// (the last is the total), optionally also narrowed to u32.
    fn scan(
        &self,
        values: &CudaSlice<u32>,
        n: u32,
        out32: Option<&mut CudaSlice<u32>>,
    ) -> Result<(CudaSlice<u64>, f32)> {
        const BLOCK: u32 = 1024;
        let blocks = (n + 1).div_ceil(BLOCK);
        let mut out = self.alloc_uninit::<u64>(n as usize + 1)?;
        let mut sums = self.alloc_uninit::<u64>(blocks as usize + 1)?;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut ms = 0.0;
        let mut a = self.stream.launch_builder(&self.scan_blocks);
        a.arg(values).arg(&n).arg(&mut out).arg(&mut sums);
        // SAFETY: `presst_scan_blocks(in, n, out, block_sums)`: reads in[0..n],
        // writes out[0..=n] and sums[0..blocks], all within their allocations.
        ms += events_ms(unsafe { a.record_kernel_launch(CUevent_flags::CU_EVENT_DEFAULT).launch(cfg) }?)?;
        let mut b = self.stream.launch_builder(&self.scan_sums);
        b.arg(&mut sums).arg(&blocks);
        let one = LaunchConfig {
            grid_dim: (1, 1, 1),
            ..cfg
        };
        // SAFETY: `presst_scan_sums(sums, nb)`: one block over sums[0..=blocks].
        ms += events_ms(unsafe { b.record_kernel_launch(CUevent_flags::CU_EVENT_DEFAULT).launch(one) }?)?;
        let mut c = self.stream.launch_builder(&self.scan_add);
        c.arg(&mut out).arg(&n).arg(&sums);
        let null: u64 = 0;
        match out32 {
            Some(o) => c.arg(o),
            None => c.arg(&null),
        };
        // SAFETY: `presst_scan_add(out, n, sums, out32)`: updates out[0..=n] and,
        // if given (a null pointer otherwise), writes out32[0..=n], which has
        // n + 1 entries.
        ms += events_ms(unsafe { c.record_kernel_launch(CUevent_flags::CU_EVENT_DEFAULT).launch(cfg) }?)?;
        Ok((out, ms))
    }

    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel signature")]
    fn gather(
        &self,
        scratch: &CudaSlice<u8>,
        ranges: &CudaSlice<u8>,
        lens: &CudaSlice<u32>,
        dst_offsets: &CudaSlice<u64>,
        n: u32,
        dst: &mut CudaSlice<u8>,
        word_stride: u32,
    ) -> Result<f32> {
        let mut g = self.stream.launch_builder(&self.gather);
        g.arg(scratch)
            .arg(ranges)
            .arg(lens)
            .arg(dst_offsets)
            .arg(&n)
            .arg(dst)
            .arg(&word_stride);
        // SAFETY: `presst_gather`: range i copies lens[i] bytes from its scratch
        // slot (words `word_stride` apart, as the encoder wrote them) to
        // dst_offsets[i]. Callers pass destination offsets that keep every copy
        // inside `dst`.
        events_ms(unsafe {
            g.record_kernel_launch(CUevent_flags::CU_EVENT_DEFAULT)
                .launch(launch_warps(n))
        }?)
    }

    /// Compresses strings that are already on the GPU (Arrow layout: `data`
    /// plus `n + 1` u32 `offsets`) into device codes and offsets. No string data
    /// or lengths pass through the host; only the total code length is read
    /// back to size the output. The result equals
    /// [`presst::Compressor::compress_strings`].
    pub fn encode_strings_device(
        &self,
        dict: &GpuDictionary,
        data: &DeviceBytes,
        offsets: &CudaSlice<u32>,
        n: usize,
    ) -> Result<(DeviceCodes, GpuTiming)> {
        let n32 = u32::try_from(n).map_err(|_| GpuError::Unavailable("too many strings for one launch"))?;
        if offsets.len() < n + 1 {
            return Err(GpuError::Unavailable("offsets must hold n + 1 entries"));
        }
        let mut ms = 0.0;
        let mut ranges = self.alloc_uninit::<u8>(n.max(1) * encode::RANGE_BYTES)?;
        let mut error = self.stream.alloc_zeros::<u32>(1)?;
        let data_len = data.len as u64;
        let mut r = self.stream.launch_builder(&self.string_ranges);
        r.arg(offsets).arg(&n32).arg(&data_len).arg(&mut ranges).arg(&mut error);
        // SAFETY: `presst_string_ranges(offsets, n, data_len, ranges, error)`: reads
        // offsets[0..=n] (the slice has at least n + 1 entries) and writes n
        // ranges and the error word, all within their allocations.
        ms += events_ms(unsafe {
            r.record_kernel_launch(CUevent_flags::CU_EVENT_DEFAULT)
                .launch(launch_1d(n32, DECODE_THREADS))
        }?)?;
        if self.stream.clone_dtoh(&error)?[0] != 0 {
            return Err(GpuError::Format(presst::Error::Corrupt(
                "string offsets are not sorted or point past the data",
            )));
        }
        // Scratch: 2 bytes per input byte plus one word per string (see the kernel).
        let scratch_len = 2 * data.len + 8 * (n + 1) + 16;
        let mut scratch = self.alloc_uninit::<u8>(scratch_len)?;
        let mut lens = self.alloc_uninit::<u32>(n.max(1))?;
        let mut e = self.stream.launch_builder(&self.encode);
        e.arg(&data.buf)
            .arg(&ranges)
            .arg(&n32)
            .arg(&dict.encode)
            .arg(&mut scratch)
            .arg(&mut lens);
        // SAFETY: `presst_encode_ranges`: the range builder checked that the offsets
        // are sorted and within `data` (the kernel reads up to 8 bytes past a
        // string, inside the slack), and every range writes
        // at most round_up(2 * len, 8) bytes into its own scratch slot, which the
        // range builder lays out without overlap inside `scratch_len`.
        ms += events_ms(unsafe {
            e.record_kernel_launch(CUevent_flags::CU_EVENT_DEFAULT)
                .launch(launch_1d(n32, DECODE_THREADS))
        }?)?;
        let mut code_offsets = self.alloc_uninit::<u32>(n + 1)?;
        let (dst_offsets, scan_ms) = self.scan(&lens, n32, Some(&mut code_offsets))?;
        ms += scan_ms;
        let mut total = [0u64];
        self.stream.memcpy_dtoh(&dst_offsets.slice(n..), &mut total)?;
        let codes_len = usize::try_from(total[0]).map_err(|_| GpuError::Unavailable("codes too large"))?;
        if u32::try_from(codes_len).is_err() {
            return Err(GpuError::Unavailable("string batch larger than 4 GiB"));
        }
        let mut codes = self.alloc_uninit::<u8>(codes_len + 16)?;
        ms += self.gather(&scratch, &ranges, &lens, &dst_offsets, n32, &mut codes, 8)?;
        Ok((
            DeviceCodes {
                codes,
                codes_len,
                offsets: code_offsets,
                len: n,
            },
            GpuTiming { kernel_ms: ms },
        ))
    }

    /// Copies device codes back into a host [`StringBatch`].
    pub fn download_codes(&self, dict: &GpuDictionary, codes: &DeviceCodes) -> Result<StringBatch> {
        let mut bytes = vec![0u8; codes.codes_len];
        if codes.codes_len > 0 {
            self.stream
                .memcpy_dtoh(&codes.codes.slice(..codes.codes_len), &mut bytes)?;
        }
        let offsets = self.stream.clone_dtoh(&codes.offsets)?;
        Ok(StringBatch::new(dict.id, bytes, offsets[..=codes.len].to_vec())?)
    }

    /// Compresses device bytes into a frame in GPU memory. Needs a split index
    /// (splits of at most 64 KiB) and no checksum: the checksum would need the
    /// content on the host. Apart from that, the frame equals
    /// [`presst::Compressor::compress_with`]. Only the code length of every
    /// split (4 bytes each) passes through the host, to lay out the frame.
    pub fn encode_frame_device(
        &self,
        dict: &GpuDictionary,
        input: &DeviceBytes,
        options: FrameOptions,
    ) -> Result<(DeviceBytes, GpuTiming)> {
        if options.checksum {
            return Err(GpuError::Unavailable(
                "device frame encoding cannot compute the checksum",
            ));
        }
        match options.split_size_log2() {
            Some(s) if s <= 16 => {}
            _ => {
                return Err(GpuError::Unavailable(
                    "device frame encoding needs splits of at most 64 KiB",
                ));
            }
        }
        let plan = encode::plan_frame(input.len, options);
        let mut header = Vec::new();
        presst::FrameHeader {
            scheme: dict.dict.scheme(),
            block_size_log2: options.block_size_log2(),
            dict_id: dict.id,
            checksum: false,
            content_size: options.content_size.then_some(input.len as u64),
            split_size_log2: options.split_size_log2(),
        }
        .write(&mut header);
        if plan.pieces.is_empty() {
            header.extend_from_slice(&block_header(true, true, 0, 0));
            return Ok((self.upload_bytes(&header)?, GpuTiming::default()));
        }
        let n = plan.pieces.len() as u32;
        let (ranges, scratch_len) = encode::ranges_interleaved(&plan.pieces);
        let ranges_dev = self.stream.clone_htod(&encode::ranges_to_bytes(&ranges))?;
        let mut scratch = self.alloc_uninit::<u8>(scratch_len as usize + 8)?;
        let mut lens_dev = self.alloc_uninit::<u32>(plan.pieces.len())?;
        let mut e = self.stream.launch_builder(&self.encode_tiled);
        e.arg(&input.buf)
            .arg(&ranges_dev)
            .arg(&n)
            .arg(&dict.encode)
            .arg(&mut scratch)
            .arg(&mut lens_dev);
        // SAFETY: `presst_encode_ranges_tiled`: pieces start at multiples of the
        // split size and are at most 64 KiB, so the tiled loads stay within the
        // input plus `DEVICE_SLACK`; every range writes inside its interleaved
        // scratch slot (`ranges_interleaved` reserves the worst case).
        let mut ms = events_ms(unsafe {
            e.record_kernel_launch(CUevent_flags::CU_EVENT_DEFAULT)
                .launch(launch_1d(n, DECODE_THREADS))
        }?)?;
        let lens = self.stream.clone_dtoh(&lens_dev)?;

        // Lay out the frame on the host: header, then per block either its
        // header + split index + codes, or its header + the raw input.
        let mut layout = FrameLayout::default();
        layout.meta(0, &header);
        let mut pos = header.len();
        let mut dst_offsets = vec![0u64; lens.len()];
        let mut gather_lens = lens.clone();
        for (b, &(start, len, first, count)) in plan.blocks.iter().enumerate() {
            let last = b + 1 == plan.blocks.len();
            let block_lens = &lens[first..first + count];
            let codes: usize = block_lens.iter().map(|&l| l as usize).sum();
            let payload = 4 * count + codes;
            if payload >= len {
                layout.meta(pos, &block_header(true, last, len, len));
                layout.raw.push((start, pos + 8, len));
                pos += 8 + len;
                gather_lens[first..first + count].fill(0);
            } else {
                let mut meta = block_header(false, last, payload, len).to_vec();
                let mut end = 0u32;
                for &l in block_lens {
                    end += l;
                    meta.extend_from_slice(&end.to_le_bytes());
                }
                layout.meta(pos, &meta);
                let mut at = (pos + meta.len()) as u64;
                for (i, &l) in block_lens.iter().enumerate() {
                    dst_offsets[first + i] = at;
                    at += u64::from(l);
                }
                pos += meta.len() + codes;
            }
        }
        // Every byte is written: headers and indexes, raw blocks, gathered codes.
        let mut frame = self.alloc_bytes_uninit(pos)?;
        for (at, bytes) in &layout.metas {
            self.stream
                .memcpy_htod(bytes, &mut frame.buf.slice_mut(*at..*at + bytes.len()))?;
        }
        for &(src, dst, len) in &layout.raw {
            self.stream.memcpy_dtod(
                &input.buf.slice(src..src + len),
                &mut frame.buf.slice_mut(dst..dst + len),
            )?;
        }
        let gather_lens_dev = self.stream.clone_htod(&gather_lens)?;
        let dst_offsets_dev = self.stream.clone_htod(&dst_offsets)?;
        ms += self.gather(
            &scratch,
            &ranges_dev,
            &gather_lens_dev,
            &dst_offsets_dev,
            n,
            &mut frame.buf,
            256,
        )?;
        Ok((frame, GpuTiming { kernel_ms: ms }))
    }
}

/// Small host-built pieces of a device frame.
#[derive(Default)]
struct FrameLayout {
    /// (frame offset, bytes): the frame header, block headers and split indexes.
    metas: Vec<(usize, Vec<u8>)>,
    /// (input offset, frame offset, length): raw blocks.
    raw: Vec<(usize, usize, usize)>,
}

impl FrameLayout {
    fn meta(&mut self, at: usize, bytes: &[u8]) {
        self.metas.push((at, bytes.to_vec()));
    }
}
