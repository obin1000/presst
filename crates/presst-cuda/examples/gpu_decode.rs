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
