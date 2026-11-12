//! Compiles `kernels/presst.cu` into a fatbin with nvcc and embeds it.
//!
//! - nvcc is taken from `NVCC`, `CUDA_PATH/bin`, or `PATH`.
//! - Target architectures come from `PRESST_CUDA_ARCHS` (comma-separated SM
//!   numbers, default below); PTX for the newest is included so newer GPUs can
//!   JIT-compile it.
//! - Without nvcc (or if it fails), an empty fatbin is embedded and
//!   `presst_cuda::is_available()` returns false, so the crate still builds and
//!   its tests skip GPU work. Set `PRESST_CUDA_REQUIRE=1` to make that an error.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Pascal (local GTX 1080) through Hopper. CUDA 12.x supports all of these.
const DEFAULT_ARCHS: &str = "61,70,75,80,86,89,90";

fn main() {
    println!("cargo:rerun-if-changed=kernels");
    for var in [
        "NVCC",
        "NVCC_CCBIN",
        "CUDA_PATH",
        "PRESST_CUDA_ARCHS",
        "PRESST_CUDA_REQUIRE",
        "PRESST_CUDA_DEBUG",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let fatbin = out.join("presst.fatbin");
    let require = env::var("PRESST_CUDA_REQUIRE").is_ok_and(|v| v == "1");
    match compile(&fatbin) {
        Ok(()) => println!("cargo:rustc-cfg=presst_cuda_kernels"),
        Err(e) => {
            assert!(
                !require,
                "PRESST_CUDA_REQUIRE=1 but the CUDA kernels could not be built: {e}"
            );
            println!("cargo:warning=presst-cuda: building without GPU kernels ({e})");
            std::fs::write(&fatbin, []).expect("write empty fatbin");
        }
    }
    println!("cargo:rustc-check-cfg=cfg(presst_cuda_kernels)");
}

fn nvcc() -> Option<PathBuf> {
    if let Ok(p) = env::var("NVCC") {
        return Some(PathBuf::from(p));
    }
    let exe = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };
    if let Ok(cuda) = env::var("CUDA_PATH") {
        let p = Path::new(&cuda).join("bin").join(exe);
        if p.exists() {
            return Some(p);
        }
    }
    env::var_os("PATH").and_then(|paths| env::split_paths(&paths).map(|d| d.join(exe)).find(|p| p.exists()))
}

fn compile(fatbin: &Path) -> Result<(), String> {
    let nvcc = nvcc().ok_or("nvcc not found (set NVCC or CUDA_PATH)")?;
    match run_nvcc(&nvcc, fatbin, false) {
        // Only device code is built, where the host compiler just preprocesses, so a
        // Visual Studio newer than this CUDA version supports is safe to allow.
        Err(e) if e.contains("unsupported Microsoft Visual Studio version") => run_nvcc(&nvcc, fatbin, true),
        r => r,
    }
}

fn run_nvcc(nvcc: &Path, fatbin: &Path, allow_unsupported_compiler: bool) -> Result<(), String> {
    let archs = env::var("PRESST_CUDA_ARCHS").unwrap_or_else(|_| DEFAULT_ARCHS.to_owned());
    let archs: Vec<&str> = archs.split(',').map(str::trim).filter(|a| !a.is_empty()).collect();
    let newest = *archs.last().ok_or("PRESST_CUDA_ARCHS is empty")?;
    let mut cmd = Command::new(nvcc);
    cmd.arg("--fatbin")
        .arg("-std=c++17")
        .arg("-O3")
        .arg("--restrict")
        .arg("-Ikernels")
        .arg("-o")
        .arg(fatbin)
        .arg("kernels/presst.cu");
    if env::var("PRESST_CUDA_DEBUG").is_ok_and(|v| v == "1") {
        cmd.arg("-lineinfo");
    }
    if allow_unsupported_compiler {
        cmd.arg("-allow-unsupported-compiler");
    }
    for a in &archs {
        cmd.arg(format!("-gencode=arch=compute_{a},code=sm_{a}"));
    }
    cmd.arg(format!("-gencode=arch=compute_{newest},code=compute_{newest}"));
    if let Ok(ccbin) = env::var("NVCC_CCBIN") {
        cmd.arg("-ccbin").arg(ccbin);
    } else if cfg!(windows) {
        // nvcc needs the MSVC host compiler. CUDA 12 supports Visual Studio up to
        // 2022, so prefer a 2022 installation (its full vcvars environment); fall
        // back to the newest compiler the cc crate finds.
        if let Some(vars) = vs2022_env() {
            cmd.envs(vars);
        } else {
            let target = env::var("TARGET").unwrap_or_default();
            if let Some(cl) = cc::windows_registry::find_tool(&target, "cl.exe") {
                if let Some(dir) = cl.path().parent() {
                    cmd.arg("-ccbin").arg(dir);
                }
                for (k, v) in cl.get_envs() {
                    cmd.env(k, v);
                }
            }
        }
    }
    let output = cmd.output().map_err(|e| format!("running {}: {e}", nvcc.display()))?;
    if !output.status.success() {
        return Err(format!(
            "nvcc failed:\n{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(())
}

/// The environment of Visual Studio 2022's `vcvars64.bat`, if installed.
fn vs2022_env() -> Option<Vec<(String, String)>> {
    let program_files = env::var("ProgramFiles(x86)").ok()?;
    let vswhere = Path::new(&program_files).join("Microsoft Visual Studio/Installer/vswhere.exe");
    let found = Command::new(vswhere)
        .args(["-version", "[17.0,18.0)", "-products", "*", "-requires"])
        .arg("Microsoft.VisualStudio.Component.VC.Tools.x86.x64")
        .args(["-property", "installationPath"])
        .output()
        .ok()?;
    let install = String::from_utf8_lossy(&found.stdout).lines().next()?.trim().to_owned();
    let vcvars = Path::new(&install).join("VC/Auxiliary/Build/vcvars64.bat");
    if !vcvars.exists() {
        return None;
    }
    let mut cmd = Command::new("cmd");
    // cmd does not understand Rust's escaping of inner quotes: pass the line as is.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt as _;
        cmd.raw_arg(format!("/c call \"{}\" >nul && set", vcvars.display()));
    }
    let out = cmd.output().ok()?;
    if !out.status.success() {
        return None;
    }
    let vars: Vec<(String, String)> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.split_once('=').map(|(k, v)| (k.to_owned(), v.to_owned())))
        .collect();
    vars.iter()
        .any(|(k, _)| k.eq_ignore_ascii_case("VCToolsInstallDir"))
        .then_some(vars)
}
