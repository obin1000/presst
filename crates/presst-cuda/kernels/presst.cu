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

// ---------------------------------------------------------------------------
// Encoding.
#include "presst_encode.cuh"

// Encode tables copied into shared memory by every thread block.
struct SharedEncodeTables {
    uint64_t long_sym[kLongSlots];
    uint8_t long_len[kLongSlots];
    uint8_t long_code[kLongSlots];
    uint32_t two[kTwoSlots];
    uint8_t byte_code[256];

    __device__ void load(const EncodeTable* __restrict__ table) {
        for (uint32_t i = threadIdx.x; i < kLongSlots; i += blockDim.x) {
            long_sym[i] = table->long_sym[i];
            long_len[i] = table->long_len[i];
            long_code[i] = table->long_code[i];
        }
        for (uint32_t i = threadIdx.x; i < kTwoSlots; i += blockDim.x) two[i] = table->two[i];
        for (uint32_t i = threadIdx.x; i < 256; i += blockDim.x) byte_code[i] = table->byte_code[i];
        __syncthreads();
    }

    __device__ EncodeTables view() const { return EncodeTables{long_sym, long_len, long_code, two, byte_code}; }
};

// One thread per range, input read from global memory (string batches). Writes
// the code length of every range to out_lens.
extern "C" __global__ void __launch_bounds__(256)
presst_encode_ranges(const uint8_t* __restrict__ in,
                     const Range* __restrict__ ranges,
                     uint32_t n_ranges,
                     const EncodeTable* __restrict__ table,
                     uint8_t* __restrict__ out,
                     uint32_t* __restrict__ out_lens) {
    __shared__ SharedEncodeTables t;
    t.load(table);
    const EncodeTables view = t.view();
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_ranges; i += gridDim.x * blockDim.x) {
        Range r = ranges[i];
        out_lens[i] = encode_range(in, r, view, out);
    }
}

// Tiled: one warp per 32 consecutive ranges, input staged through shared
// memory (frames: range starts are multiples of the split size).
extern "C" __global__ void __launch_bounds__(256)
presst_encode_ranges_tiled(const uint8_t* __restrict__ in,
                           const Range* __restrict__ ranges,
                           uint32_t n_ranges,
                           const EncodeTable* __restrict__ table,
                           uint8_t* __restrict__ out,
                           uint32_t* __restrict__ out_lens) {
    __shared__ SharedEncodeTables t;
    __shared__ __align__(16) uint8_t rows[8][32 * kRowBytes];
    t.load(table);
    const EncodeTables view = t.view();
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t warps = gridDim.x * 8;
    for (uint32_t first = (blockIdx.x * 8 + warp) * 32; first < n_ranges; first += warps * 32) {
        const uint32_t i = first + lane;
        Range r = i < n_ranges ? ranges[i] : Range{0, 0, 0, 0};
        if (i >= n_ranges) r.in_start = ranges[first].in_start;  // a valid address to load from
        const uint32_t n = encode_range_tiled(in, r, view, out, rows[warp]);
        if (i < n_ranges) out_lens[i] = n;
    }
}

// Compaction: one warp per range copies its codes from the scratch area to
// dst[dst_offsets[i] ..]. Word j of a range's codes is at out_start + j * word_stride
// (8: contiguous, 256: interleaved by the tiled encoder).
extern "C" __global__ void __launch_bounds__(256)
presst_gather(const uint8_t* __restrict__ scratch,
              const Range* __restrict__ ranges,
              const uint32_t* __restrict__ lens,
              const uint64_t* __restrict__ dst_offsets,
              uint32_t n_ranges,
              uint8_t* __restrict__ dst,
              uint32_t word_stride) {
    const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t warps = gridDim.x * blockDim.x / 32;
    for (uint32_t i = warp; i < n_ranges; i += warps) {
        const uint8_t* src = scratch + ranges[i].out_start;
        uint8_t* d = dst + dst_offsets[i];
        const uint32_t n = lens[i];
        for (uint32_t j = lane; j < n; j += 32) d[j] = src[(j >> 3) * word_stride + (j & 7)];
    }
}
