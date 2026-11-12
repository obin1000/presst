# Vortex FSST kernels

- `fsst_warp.cu`: the warp-per-32-strings FSST decoder for Vortex, written for
  presst and vendored into the Vortex fork (`vortex-cuda/kernels/src/fsst_warp.cu`,
  branch `presst` of https://github.com/obin1000/vortex). This copy is the
  source of truth; keep the two identical.
- `fsst.cu`, `config.cuh`: Vortex's own thread-per-string FSST decoder, copied
  unchanged from Vortex 0.93.0 (Apache-2.0, see the SPDX headers) so the
  benchmarks can compare both kernels on the same GPU and data.

Both are compiled into presst-cuda's fatbin through `presst.cu`.
