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

// K2: one warp per split (frames) or per group of 32 strings (string batches).
// kSharedTable selects where the decode table is read from.
template <bool kSharedTable, bool kStrings>
__device__ __forceinline__ void decode_warp(const uint8_t* __restrict__ in,
                                            const Split* __restrict__ splits,
                                            uint32_t n_splits,
                                            const DecodeTable* __restrict__ table,
                                            uint8_t* __restrict__ out,
                                            uint32_t* __restrict__ error,
                                            const uint32_t* __restrict__ code_offsets,
                                            uint32_t n_strings,
                                            uint32_t* __restrict__ out_offsets) {
    constexpr uint32_t kWarpsPerBlock = 256 / kWarp;
    __shared__ __align__(16) uint8_t stage[kWarpsPerBlock][kStageBytes];
    __shared__ uint64_t sym[kSharedTable ? 256 : 1];
    __shared__ uint8_t len[kSharedTable ? 256 : 1];
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
        StringGroup g{code_offsets, i * kWarp, kStrings ? min(kWarp, n_strings - i * kWarp) : 0u, out_offsets};
        uint32_t e = kSharedTable
                         ? decode_range_warp<true, kStrings>(in, s, sym, len, out, stage[warp], g)
                         : decode_range_warp<false, kStrings>(in, s, table->sym, table->len, out, stage[warp], g);
        if (e && lane == 0) atomicOr(error, e);
    }
}

#define FRAME_ARGS                                                                        \
    const uint8_t *__restrict__ in, const Split *__restrict__ splits, uint32_t n_splits, \
        const DecodeTable *__restrict__ table, uint8_t *__restrict__ out, uint32_t *__restrict__ error
#define STRING_ARGS \
    FRAME_ARGS, const uint32_t *__restrict__ code_offsets, uint32_t n_strings, uint32_t *__restrict__ out_offsets

extern "C" __global__ void __launch_bounds__(256) presst_decode_warp_shared(FRAME_ARGS) {
    decode_warp<true, false>(in, splits, n_splits, table, out, error, nullptr, 0, nullptr);
}

extern "C" __global__ void __launch_bounds__(256) presst_decode_warp_global(FRAME_ARGS) {
    decode_warp<false, false>(in, splits, n_splits, table, out, error, nullptr, 0, nullptr);
}

// String batches: split i is group i, holding strings 32i .. 32i + 31.
extern "C" __global__ void __launch_bounds__(256) presst_decode_strings_shared(STRING_ARGS) {
    decode_warp<true, true>(in, splits, n_splits, table, out, error, code_offsets, n_strings, out_offsets);
}

extern "C" __global__ void __launch_bounds__(256) presst_decode_strings_global(STRING_ARGS) {
    decode_warp<false, true>(in, splits, n_splits, table, out, error, code_offsets, n_strings, out_offsets);
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
