//! `presst` command-line tool.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use presst::io::{FrameReader, FrameWriter};
use presst::{Compressor, Decompressor, Dictionary, FrameHeader, FrameOptions, StringBatch};
use presst_train::{Corpus, Mode, Split, TrainOptions, train};

#[derive(Parser)]
#[command(name = "presst", version, about = "Pretrained static symbol table compression")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train a dictionary on sample files.
    Train(TrainArgs),
    /// Compress a file into a frame.
    Compress(CodecArgs),
    /// Decompress a frame.
    Decompress(CodecArgs),
    /// Show information about a dictionary, frame or string batch.
    Inspect {
        /// File to inspect.
        file: PathBuf,
        /// Also list every symbol of a dictionary.
        #[arg(long)]
        symbols: bool,
    },
    /// Measure ratio and single-threaded speed of a dictionary on files.
    Eval {
        /// Dictionary to use.
        #[arg(short, long)]
        dict: PathBuf,
        /// Files to evaluate on (use data the dictionary was not trained on).
        #[arg(required = true)]
        files: Vec<PathBuf>,
        /// Treat each line as a separate string (column mode) instead of one stream.
        #[arg(long)]
        lines: bool,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum ModeArg {
    /// FSST's own algorithm (16 KiB sample, 5 generations): the baseline.
    Fsst,
    /// Large sample, many generations, local search.
    Extended,
}

#[derive(clap::Args)]
struct TrainArgs {
    /// Sample files.
    #[arg(required = true)]
    files: Vec<PathBuf>,
    /// Output dictionary.
    #[arg(short, long)]
    output: PathBuf,
    /// Treat each line as a separate record (for string columns).
    #[arg(long, conflicts_with = "window")]
    lines: bool,
    /// Record size in bytes when not using --lines.
    #[arg(long, default_value_t = 4096)]
    window: usize,
    /// Training mode.
    #[arg(long, value_enum, default_value_t = ModeArg::Extended)]
    mode: ModeArg,
    /// Random seed.
    #[arg(long)]
    seed: Option<u64>,
    /// Training sample size in MiB.
    #[arg(long)]
    sample_mib: Option<f64>,
    /// Maximum number of generations (extended mode).
    #[arg(long)]
    generations: Option<usize>,
    /// Local search steps (extended mode, 0 disables).
    #[arg(long)]
    local_search: Option<usize>,
    /// Number of threads (default: all cores).
    #[arg(long)]
    threads: Option<usize>,
    /// Name stored in the dictionary.
    #[arg(long)]
    name: Option<String>,
}

#[derive(clap::Args)]
struct CodecArgs {
    /// Dictionary to use.
    #[arg(short, long)]
    dict: PathBuf,
    /// Input file, or "-" for stdin.
    input: PathBuf,
    /// Output file, or "-" for stdout (default: stdout).
    #[arg(short, long, default_value = "-")]
    output: PathBuf,
    /// Block size as a power of two (compress only).
    #[arg(long, default_value_t = presst::DEFAULT_BLOCK_SIZE_LOG2)]
    block_size_log2: u8,
    /// Omit the content checksum (compress only).
    #[arg(long)]
    no_checksum: bool,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Train(args) => cmd_train(&args),
        Command::Compress(args) => cmd_compress(&args),
        Command::Decompress(args) => cmd_decompress(&args),
        Command::Inspect { file, symbols } => cmd_inspect(&file, symbols),
        Command::Eval { dict, files, lines } => cmd_eval(&dict, &files, lines),
    }
}

fn load_dict(path: &Path) -> Result<Dictionary> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Dictionary::from_bytes(&bytes).with_context(|| format!("parsing dictionary {}", path.display()))
}

fn open_input(path: &Path) -> Result<Box<dyn Read>> {
    Ok(if path.as_os_str() == "-" {
        Box::new(io::stdin().lock())
    } else {
        Box::new(BufReader::new(
            File::open(path).with_context(|| format!("opening {}", path.display()))?,
        ))
    })
}

fn open_output(path: &Path) -> Result<Box<dyn Write>> {
    Ok(if path.as_os_str() == "-" {
        Box::new(io::stdout().lock())
    } else {
        Box::new(BufWriter::new(
            File::create(path).with_context(|| format!("creating {}", path.display()))?,
        ))
    })
}

fn cmd_train(args: &TrainArgs) -> Result<()> {
    let split = if args.lines {
        Split::Lines
    } else {
        Split::Windows(args.window)
    };
    let mut corpus = Corpus::new();
    for f in &args.files {
        let data = std::fs::read(f).with_context(|| format!("reading {}", f.display()))?;
        corpus.add(&data, split);
    }
    let mut opts = match args.mode {
        ModeArg::Fsst => TrainOptions::fsst(),
        ModeArg::Extended => TrainOptions::extended(),
    };
    if let Some(s) = args.seed {
        opts.seed = s;
    }
    if let Some(mib) = args.sample_mib {
        opts.sample_bytes = (mib * 1024.0 * 1024.0) as usize;
    }
    if let Some(g) = args.generations {
        opts.max_generations = g;
    }
    if let Some(l) = args.local_search {
        opts.local_search_steps = l;
    }
    opts.threads = args.threads;
    opts.name.clone_from(&args.name);

    eprintln!(
        "training on {} records ({:.1} MiB), mode {}",
        corpus.len(),
        corpus.total_bytes() as f64 / 1_048_576.0,
        if opts.mode == Mode::Fsst { "fsst" } else { "extended" }
    );
    let start = Instant::now();
    let report = train(&corpus, &opts)?;
    for g in &report.generations {
        eprintln!(
            "  generation {:>3}: {:>3} symbols, validation ratio {:.3}",
            g.generation, g.symbols, g.validation_ratio
        );
    }
    if report.local_search_accepted > 0 {
        eprintln!("  local search: {} swaps accepted", report.local_search_accepted);
    }
    eprintln!(
        "done in {:.1}s: {} symbols, validation ratio {:.3} (sample {:.1} MiB, validation {:.1} MiB)",
        start.elapsed().as_secs_f64(),
        report.dictionary.len(),
        report.validation_ratio,
        report.sample_bytes as f64 / 1_048_576.0,
        report.validation_bytes as f64 / 1_048_576.0
    );
    std::fs::write(&args.output, report.dictionary.to_bytes())
        .with_context(|| format!("writing {}", args.output.display()))?;
    println!("{}", report.dictionary.id());
    Ok(())
}

fn cmd_compress(args: &CodecArgs) -> Result<()> {
    let dict = load_dict(&args.dict)?;
    let opts = FrameOptions {
        block_size_log2: args.block_size_log2,
        checksum: !args.no_checksum,
        content_size: false,
    };
    let mut input = open_input(&args.input)?;
    let mut writer = FrameWriter::new(Compressor::new(&dict)?, open_output(&args.output)?, opts);
    io::copy(&mut input, &mut writer)?;
    writer.finish()?.flush()?;
    Ok(())
}

fn cmd_decompress(args: &CodecArgs) -> Result<()> {
    let dict = load_dict(&args.dict)?;
    let mut reader = FrameReader::new(Decompressor::new(&dict)?, open_input(&args.input)?)?;
    let mut output = open_output(&args.output)?;
    io::copy(&mut reader, &mut output)?;
    output.flush()?;
    Ok(())
}

fn printable(sym: &[u8]) -> String {
    sym.iter()
        .flat_map(|&b| std::ascii::escape_default(b))
        .map(char::from)
        .collect()
}

fn cmd_inspect(path: &Path, list_symbols: bool) -> Result<()> {
    let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    match data.get(..4) {
        Some(b"PRSD") => {
            let d = Dictionary::from_bytes(&data)?;
            let m = d.metadata();
            println!("dictionary {}", d.id());
            println!("  scheme:   {}", d.scheme());
            println!("  symbols:  {}", d.len());
            let lens = d.symbol_lens();
            for l in 1..=d.max_symbol_len() {
                let n = lens.iter().filter(|&&x| x as usize == l).count();
                if n > 0 {
                    println!("    length {l}: {n}");
                }
            }
            if let Some(n) = &m.name {
                println!("  name:     {n}");
            }
            if let Some(t) = &m.trainer {
                println!("  trainer:  {t}");
            }
            if let Some(s) = m.seed {
                println!("  seed:     {s}");
            }
            if let Some(h) = m.corpus_hash {
                println!("  corpus:   {h:016x}");
            }
            if list_symbols {
                for (code, s) in d.symbols().enumerate() {
                    println!("  {code:>3}  \"{}\"", printable(s));
                }
            }
        }
        Some(b"PRST") => {
            let (h, header_len) = FrameHeader::parse(&data)?;
            println!("frame");
            println!("  dictionary: {}", h.dict_id);
            println!("  scheme:     {}", h.scheme);
            println!("  block size: {} bytes", h.block_size());
            println!("  checksum:   {}", h.checksum);
            if let Some(s) = h.content_size {
                println!("  content:    {s} bytes");
            }
            let (mut pos, mut blocks, mut raw, mut content) = (header_len, 0usize, 0usize, 0u64);
            while pos + 8 <= data.len() {
                let word = u32::from_le_bytes(data[pos..pos + 4].try_into()?);
                let dsize = u32::from_le_bytes(data[pos + 4..pos + 8].try_into()?);
                blocks += 1;
                raw += usize::from(word >> 30 == 1);
                content += u64::from(dsize);
                pos += 8 + (word & ((1 << 29) - 1)) as usize;
                if word & (1 << 29) != 0 {
                    break;
                }
            }
            println!("  blocks:     {blocks} ({raw} raw)");
            println!("  ratio:      {:.3}", content as f64 / data.len().max(1) as f64);
        }
        Some(b"PRSC") => {
            let b = StringBatch::from_bytes(&data)?;
            println!("string batch");
            println!("  dictionary: {}", b.dict_id());
            println!("  strings:    {}", b.len());
            println!("  code bytes: {}", b.codes_len());
        }
        _ => bail!("{} is not a presst dictionary, frame or string batch", path.display()),
    }
    Ok(())
}

fn cmd_eval(dict_path: &Path, files: &[PathBuf], lines: bool) -> Result<()> {
    let dict = load_dict(dict_path)?;
    let comp = Compressor::new(&dict)?;
    let dec = Decompressor::new(&dict)?;
    println!("dictionary {} ({} symbols)", dict.id(), dict.len());
    println!(
        "{:<40} {:>12} {:>8} {:>12} {:>12}",
        "file", "bytes", "ratio", "comp MB/s", "decomp MB/s"
    );
    for f in files {
        let data = std::fs::read(f).with_context(|| format!("reading {}", f.display()))?;
        let name = f
            .file_name()
            .map_or_else(|| f.display().to_string(), |n| n.to_string_lossy().into_owned());
        let (ratio, cs, ds) = if lines {
            let strings: Vec<&[u8]> = data.split(|&b| b == b'\n').collect();
            let input: usize = strings.iter().map(|s| s.len()).sum();
            let (batch, ct) = time(|| comp.compress_strings(&strings));
            let (_, dt) = time(|| dec.decompress_strings(&batch).expect("roundtrip"));
            let encoded = batch.codes_len() + 4 * batch.len();
            (input as f64 / encoded.max(1) as f64, mbps(input, ct), mbps(input, dt))
        } else {
            let (frame, ct) = time(|| comp.compress(&data));
            let (out, dt) = time(|| dec.decompress(&frame).expect("roundtrip"));
            assert_eq!(out, data, "roundtrip mismatch");
            (
                data.len() as f64 / frame.len() as f64,
                mbps(data.len(), ct),
                mbps(data.len(), dt),
            )
        };
        println!("{name:<40} {:>12} {ratio:>8.3} {cs:>12.0} {ds:>12.0}", data.len());
    }
    if lines {
        println!("(column mode: ratio includes 4-byte offsets per string)");
    }
    Ok(())
}

/// Runs `f` repeatedly for at least 0.3 s and returns its result and the
/// best time per run in seconds.
fn time<T>(mut f: impl FnMut() -> T) -> (T, f64) {
    let start = Instant::now();
    let mut best = f64::INFINITY;
    let mut out;
    loop {
        let t = Instant::now();
        out = f();
        best = best.min(t.elapsed().as_secs_f64());
        if start.elapsed().as_secs_f64() > 0.3 {
            break;
        }
    }
    (out, best)
}

fn mbps(bytes: usize, secs: f64) -> f64 {
    bytes as f64 / 1e6 / secs.max(1e-12)
}
