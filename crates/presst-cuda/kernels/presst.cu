// Kernel entry points for presst-cuda. The device code lives in the headers.
#include "presst_decode.cuh"

using namespace presst;

// K1: one thread per split, decode table in shared memory.
extern "C" __global__ void __launch_bounds__(256)
presst_decode_splits_shared(const uint8_t* __restrict__ in,
                            const Split* __restrict__ splits,
                            uint32_t n_splits,
                            const DecodeTable* __restrict__ table,
                            uint8_t* __restrict__ out,
                            uint32_t* __restrict__ error) {
    __shared__ uint64_t sym[256];
    __shared__ uint8_t len[256];
    for (uint32_t i = threadIdx.x; i < 256; i += blockDim.x) {
        sym[i] = table->sym[i];
        len[i] = table->len[i];
    }
    __syncthreads();
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_splits; i += gridDim.x * blockDim.x) {
        Split s = splits[i];
        uint32_t e = decode_split<true>(in, s, sym, len, out);
        if (e) {
            atomicOr(error, e);
        }
    }
}

// K2: one warp per split. kSharedTable selects where the decode table is read from.
template <bool kSharedTable>
__device__ __forceinline__ void decode_splits_warp(const uint8_t* __restrict__ in,
                                                   const Split* __restrict__ splits,
                                                   uint32_t n_splits,
                                                   const DecodeTable* __restrict__ table,
                                                   uint8_t* __restrict__ out,
                                                   uint32_t* __restrict__ error) {
    constexpr uint32_t kWarpsPerBlock = 256 / kWarp;
    __shared__ __align__(16) uint8_t stage[kWarpsPerBlock][kStageBytes];
    __shared__ uint64_t sym[256];
    __shared__ uint8_t len[256];
    if (kSharedTable) {
        for (uint32_t i = threadIdx.x; i < 256; i += blockDim.x) {
            sym[i] = table->sym[i];
            len[i] = table->len[i];
        }
        __syncthreads();
    }
    const uint32_t warp = threadIdx.x / kWarp;
    const uint32_t lane = threadIdx.x % kWarp;
    for (uint32_t i = blockIdx.x * kWarpsPerBlock + warp; i < n_splits; i += gridDim.x * kWarpsPerBlock) {
        Split s = splits[i];
        uint32_t e = kSharedTable ? decode_split_warp<true>(in, s, sym, len, out, stage[warp])
                                  : decode_split_warp<false>(in, s, table->sym, table->len, out, stage[warp]);
        if (e && lane == 0) atomicOr(error, e);
    }
}

extern "C" __global__ void __launch_bounds__(256)
presst_decode_warp_shared(const uint8_t* __restrict__ in, const Split* __restrict__ splits, uint32_t n_splits,
                          const DecodeTable* __restrict__ table, uint8_t* __restrict__ out,
                          uint32_t* __restrict__ error) {
    decode_splits_warp<true>(in, splits, n_splits, table, out, error);
}

extern "C" __global__ void __launch_bounds__(256)
presst_decode_warp_global(const uint8_t* __restrict__ in, const Split* __restrict__ splits, uint32_t n_splits,
                          const DecodeTable* __restrict__ table, uint8_t* __restrict__ out,
                          uint32_t* __restrict__ error) {
    decode_splits_warp<false>(in, splits, n_splits, table, out, error);
}

// K1 with the table read through the read-only (L1) cache instead.
extern "C" __global__ void __launch_bounds__(256)
presst_decode_splits_global(const uint8_t* __restrict__ in,
                            const Split* __restrict__ splits,
                            uint32_t n_splits,
                            const DecodeTable* __restrict__ table,
                            uint8_t* __restrict__ out,
                            uint32_t* __restrict__ error) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_splits; i += gridDim.x * blockDim.x) {
        Split s = splits[i];
        uint32_t e = decode_split<false>(in, s, table->sym, table->len, out);
        if (e) {
            atomicOr(error, e);
        }
    }
}
