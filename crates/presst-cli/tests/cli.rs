//! End-to-end test of the `presst` binary: train, compress, decompress, inspect, eval.

use std::fmt::Write as _;
use std::process::Command;

fn presst(args: &[&std::ffi::OsStr]) -> std::process::Output {
    let out = Command::new(env!("CARGO_BIN_EXE_presst"))
        .args(args)
        .output()
        .expect("run presst");
    assert!(
        out.status.success(),
        "presst {args:?} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    out
}

#[test]
fn train_compress_decompress() {
    let dir = tempfile::tempdir().unwrap();
    let p = |name: &str| dir.path().join(name);
    let mut text = String::new();
    for i in 0..3000 {
        writeln!(text, "https://example.com/products/{i}/details?ref=home&lang=en").unwrap();
    }
    std::fs::write(p("train.txt"), &text).unwrap();
    std::fs::write(p("test.txt"), text.replace("products", "items")).unwrap();

    let id = presst(&[
        "train".as_ref(),
        p("train.txt").as_os_str(),
        "--lines".as_ref(),
        "--local-search".as_ref(),
        "20".as_ref(),
        "-o".as_ref(),
        p("d.pdict").as_os_str(),
    ]);
    let id = String::from_utf8(id.stdout).unwrap();
    assert_eq!(id.trim().len(), 16, "prints the dictionary id");

    presst(&[
        "compress".as_ref(),
        "-d".as_ref(),
        p("d.pdict").as_os_str(),
        p("test.txt").as_os_str(),
        "-o".as_ref(),
        p("test.prs").as_os_str(),
    ]);
    presst(&[
        "decompress".as_ref(),
        "-d".as_ref(),
        p("d.pdict").as_os_str(),
        p("test.prs").as_os_str(),
        "-o".as_ref(),
        p("test.out").as_os_str(),
    ]);
    let original = std::fs::read(p("test.txt")).unwrap();
    let compressed = std::fs::read(p("test.prs")).unwrap();
    assert_eq!(std::fs::read(p("test.out")).unwrap(), original);
    assert!(compressed.len() * 2 < original.len(), "compresses well");

    let info = presst(&["inspect".as_ref(), p("test.prs").as_os_str()]);
    assert!(String::from_utf8_lossy(&info.stdout).contains(id.trim()));
    let info = presst(&["inspect".as_ref(), p("d.pdict").as_os_str(), "--symbols".as_ref()]);
    assert!(String::from_utf8_lossy(&info.stdout).contains("255"));
    presst(&[
        "eval".as_ref(),
        "-d".as_ref(),
        p("d.pdict").as_os_str(),
        p("test.txt").as_os_str(),
        "--lines".as_ref(),
    ]);
}
