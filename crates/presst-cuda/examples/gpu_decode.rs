//! Decode throughput on the GPU vs. the CPU for one dictionary and input.
//!
//! ```sh
//! cargo run --release -p presst-cuda --example gpu_decode -- <dict.pdict> <input> [split_log2] [runs]
//! cargo run --release -p presst-cuda --example gpu_decode -- <dict.pdict> <input.vals> strings [runs]
//! ```
//!
//! `input` is either raw bytes or a `.vals` file (u32 length-prefixed values,
//! joined with newlines). With `strings`, every value of the `.vals` file is
//! compressed as one string of a batch with a decoded index (format v2), and
//! `decompress_strings` is measured instead. Reports kernel-only GB/s of
//! decoded bytes, end-to-end GB/s including `PCIe` transfers, and the CPU.

use std::time::Instant;

use presst::{Compressor, Decompressor, Dictionary, FrameOptions, StringOptions};
use presst_cuda::{DecodeKernel, GpuContext};

fn read_input(path: &str) -> Vec<u8> {
    let data = std::fs::read(path).expect("read input");
    if !std::path::Path::new(path)
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("vals"))
    {
        return data;
    }
    let mut out = Vec::with_capacity(data.len());
    let mut pos = 0;
    while pos + 4 <= data.len() {
        let n = u32::from_le_bytes(data[pos..pos + 4].try_into().expect("4 bytes")) as usize;
        pos += 4;
        out.extend_from_slice(&data[pos..pos + n]);
        out.push(b'\n');
        pos += n;
    }
    out
}

fn read_values(path: &str) -> Vec<Vec<u8>> {
    let data = std::fs::read(path).expect("read input");
    let mut out = Vec::new();
    let mut pos = 0;
    while pos + 4 <= data.len() {
        let n = u32::from_le_bytes(data[pos..pos + 4].try_into().expect("4 bytes")) as usize;
        pos += 4;
        out.push(data[pos..pos + n].to_vec());
        pos += n;
    }
    out
}

fn strings(dict: &Dictionary, path: &str, runs: usize) -> Result<(), Box<dyn std::error::Error>> {
    let values = read_values(path);
    let bytes: usize = values.iter().map(Vec::len).sum();
    let batch = Compressor::new(dict)?.compress_strings_with(&values, StringOptions { decoded_index: true });
    let gb = bytes as f64 / 1e9;
    println!(
        "{} strings, {:.1} MB (avg {:.1} B), codes {:.1} MB (ratio {:.2})",
        values.len(),
        bytes as f64 / 1e6,
        bytes as f64 / values.len().max(1) as f64,
        batch.codes_len() as f64 / 1e6,
        bytes as f64 / batch.codes_len().max(1) as f64
    );
    let gpu = GpuContext::new(0)?;
    println!("GPU: {}", gpu.device_name()?);
    let table = gpu.upload(dict)?;
    let options = StringOptions { decoded_index: true };
    let mut best_kernel = f64::MAX;
    let mut best_total = f64::MAX;
    for _ in 0..runs {
        let start = Instant::now();
        let (gpu_batch, timing) = gpu.compress_strings_timed(&table, &values, options)?;
        best_total = best_total.min(start.elapsed().as_secs_f64());
        best_kernel = best_kernel.min(f64::from(timing.kernel_ms) / 1e3);
        assert_eq!(gpu_batch, batch);
    }
    println!(
        "GPU strings encode: kernel {:.1} GB/s, end-to-end {:.2} GB/s",
        gb / best_kernel,
        gb / best_total
    );
    let comp = Compressor::new(dict)?;
    let mut best = f64::MAX;
    for _ in 0..runs.min(5) {
        let start = Instant::now();
        let b = comp.compress_strings_with(&values, options);
        best = best.min(start.elapsed().as_secs_f64());
        assert_eq!(b.len(), values.len());
    }
    println!("CPU strings encode (1 thread): {:.2} GB/s", gb / best);

    // Device-resident: data uploaded once, results stay on the GPU.
    let mut data = Vec::with_capacity(bytes);
    let mut offsets = vec![0u32];
    for v in &values {
        data.extend_from_slice(v);
        offsets.push(data.len() as u32);
    }
    let data_dev = gpu.upload_bytes(&data)?;
    let offsets_dev = gpu.stream().clone_htod(&offsets)?;
    let mut best = f64::MAX;
    for _ in 0..runs {
        let start = Instant::now();
        let (codes, _) = gpu.encode_strings_device(&table, &data_dev, &offsets_dev, values.len())?;
        gpu.stream().synchronize()?;
        best = best.min(start.elapsed().as_secs_f64());
        assert_eq!(codes.codes_len, batch.codes_len());
    }
    println!("GPU strings encode, device-resident: {:.1} GB/s", gb / best);
    let uploaded = gpu.upload_strings(&table, &batch)?;
    let mut best = f64::MAX;
    for _ in 0..runs {
        let start = Instant::now();
        let (out, _) = gpu.decode_strings_device(&table, &uploaded, DecodeKernel::WarpShared)?;
        gpu.stream().synchronize()?;
        best = best.min(start.elapsed().as_secs_f64());
        assert_eq!(out.data.len(), bytes);
    }
    println!("GPU strings decode, device-resident: {:.1} GB/s", gb / best);
    for kernel in [DecodeKernel::WarpShared, DecodeKernel::WarpGlobal] {
        let mut best_kernel = f64::MAX;
        let mut best_total = f64::MAX;
        for _ in 0..runs {
            let start = Instant::now();
            let ((data, offsets), timing) = gpu.decompress_strings_timed(&table, &batch, kernel)?;
            best_total = best_total.min(start.elapsed().as_secs_f64());
            best_kernel = best_kernel.min(f64::from(timing.kernel_ms) / 1e3);
            assert_eq!(data.len(), bytes);
            assert_eq!(offsets.len(), values.len() + 1);
        }
        println!(
            "GPU strings {kernel:?}: kernel {:.1} GB/s, end-to-end {:.2} GB/s",
            gb / best_kernel,
            gb / best_total
        );
    }
    let dec = Decompressor::new(dict)?;
    let mut best = f64::MAX;
    for _ in 0..runs.min(5) {
        let start = Instant::now();
        let (data, _) = dec.decompress_strings(&batch)?;
        best = best.min(start.elapsed().as_secs_f64());
        assert_eq!(data.len(), bytes);
    }
    println!("CPU strings (1 thread): {:.2} GB/s", gb / best);
    Ok(())
}

#[expect(clippy::too_many_lines, reason = "a linear benchmark script")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: gpu_decode <dict.pdict> <input> [split_log2] [runs]");
        std::process::exit(2);
    }
    let dict = Dictionary::from_bytes(&std::fs::read(&args[0])?)?;
    if args.get(2).map(String::as_str) == Some("strings") {
        let runs: usize = args.get(3).map_or(Ok(10), |s| s.parse())?;
        return strings(&dict, &args[1], runs);
    }
    let input = read_input(&args[1]);
    let split: u8 = args.get(2).map_or(Ok(12), |s| s.parse())?;
    let runs: usize = args.get(3).map_or(Ok(10), |s| s.parse())?;

    let frame = Compressor::new(&dict)?.compress_with(
        &input,
        FrameOptions {
            split_size_log2: Some(split),
            checksum: false,
            ..FrameOptions::default()
        },
    );
    let gb = input.len() as f64 / 1e9;
    println!(
        "input {:.1} MB, frame {:.1} MB (ratio {:.2}), split {} B",
        input.len() as f64 / 1e6,
        frame.len() as f64 / 1e6,
        input.len() as f64 / frame.len() as f64,
        1 << split
    );

    let gpu = GpuContext::new(0)?;
    println!("GPU: {}", gpu.device_name()?);
    let table = gpu.upload(&dict)?;
    let options = FrameOptions {
        split_size_log2: Some(split),
        checksum: false,
        ..FrameOptions::default()
    };
    let mut best_kernel = f64::MAX;
    let mut best_total = f64::MAX;
    for _ in 0..runs {
        let start = Instant::now();
        let (gpu_frame, timing) = gpu.compress_frame_timed(&table, &input, options)?;
        best_total = best_total.min(start.elapsed().as_secs_f64());
        best_kernel = best_kernel.min(f64::from(timing.kernel_ms) / 1e3);
        assert_eq!(gpu_frame, frame);
    }
    println!(
        "GPU encode: kernel {:.1} GB/s, end-to-end {:.2} GB/s",
        gb / best_kernel,
        gb / best_total
    );
    let comp = Compressor::new(&dict)?;
    let mut best = f64::MAX;
    for _ in 0..runs.min(5) {
        let start = Instant::now();
        let f = comp.compress_with(&input, options);
        best = best.min(start.elapsed().as_secs_f64());
        assert_eq!(f.len(), frame.len());
    }
    println!("CPU encode (1 thread): {:.2} GB/s", gb / best);

    // Device-resident: data uploaded once, results stay on the GPU.
    let input_dev = gpu.upload_bytes(&input)?;
    let mut best = f64::MAX;
    for _ in 0..runs {
        let start = Instant::now();
        let (f, _) = gpu.encode_frame_device(&table, &input_dev, options)?;
        gpu.stream().synchronize()?;
        best = best.min(start.elapsed().as_secs_f64());
        assert_eq!(f.len(), frame.len());
    }
    println!("GPU encode, device-resident: {:.1} GB/s", gb / best);
    let uploaded = gpu.upload_frame(&frame)?;
    let mut best = f64::MAX;
    for _ in 0..runs {
        let start = Instant::now();
        let (out, _) = gpu.decode_frame_device(&table, &uploaded, DecodeKernel::WarpShared)?;
        gpu.stream().synchronize()?;
        best = best.min(start.elapsed().as_secs_f64());
        assert_eq!(out.len(), input.len());
    }
    println!("GPU decode, device-resident: {:.1} GB/s", gb / best);
    for placement in DecodeKernel::ALL {
        let mut best_kernel = f64::MAX;
        let mut best_total = f64::MAX;
        for _ in 0..runs {
            let start = Instant::now();
            let (out, timing) = gpu.decompress_frame_timed(&table, &frame, placement)?;
            best_total = best_total.min(start.elapsed().as_secs_f64());
            best_kernel = best_kernel.min(f64::from(timing.kernel_ms) / 1e3);
            assert_eq!(out.len(), input.len());
        }
        println!(
            "GPU {placement:?}: kernel {:.1} GB/s, end-to-end {:.2} GB/s",
            gb / best_kernel,
            gb / best_total
        );
    }

    let dec = Decompressor::new(&dict)?;
    let mut best = f64::MAX;
    for _ in 0..runs.min(5) {
        let start = Instant::now();
        let out = dec.decompress(&frame)?;
        best = best.min(start.elapsed().as_secs_f64());
        assert_eq!(out.len(), input.len());
    }
    println!(
        "CPU (1 frame, {} blocks): {:.2} GB/s",
        input.len().div_ceil(1 << 20),
        gb / best
    );
    Ok(())
}
