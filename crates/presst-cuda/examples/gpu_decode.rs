//! Decode throughput on the GPU vs. the CPU for one dictionary and input.
//!
//! ```sh
//! cargo run --release -p presst-cuda --example gpu_decode -- <dict.pdict> <input> [split_log2] [runs]
//! ```
//!
//! `input` is either raw bytes or a `.vals` file (u32 length-prefixed values,
//! joined with newlines). Reports kernel-only GB/s of decoded bytes for each
//! table placement, end-to-end GB/s including PCIe transfers, and the CPU.

use std::time::Instant;

use presst::{Compressor, Decompressor, Dictionary, FrameOptions};
use presst_cuda::{DecodeKernel, GpuContext};

fn read_input(path: &str) -> Vec<u8> {
    let data = std::fs::read(path).expect("read input");
    if !path.ends_with(".vals") {
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: gpu_decode <dict.pdict> <input> [split_log2] [runs]");
        std::process::exit(2);
    }
    let dict = Dictionary::from_bytes(&std::fs::read(&args[0])?)?;
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
