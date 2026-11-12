// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// FSST decompression with one warp per group of 32 strings (from presst's
// GPU decoder, https://github.com/obin1000/presst).
//
// The warp decodes the group's concatenated code stream 128 codes at a time
// (4 per lane): coalesced code loads, escapes resolved by a warp scan over
// 2-bit carry maps, output positions by a warp prefix sum, bytes staged in
// shared memory and drained as aligned 16-byte stores. The per-string output
// offsets are given (prefix sums of the uncompressed lengths), so only the
// group's start and size are needed. Afterwards every lane writes the view of
// its string.
//
// The symbol table is copied into shared memory per block; codes at or above
// `num_symbols` decode as empty, so no read goes past the table.
//
// Null strings are expected to have empty code streams and zero length, as
// Vortex's FSST compressor writes them; a group where that does not hold is
// decoded one string per lane instead.

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr uint32_t kWarp = 32;
constexpr uint32_t kWarpsPerBlock = 8;  // blocks of 256 threads
constexpr uint32_t kCodesPerLane = 4;
constexpr uint32_t kCodesPerStep = kWarp * kCodesPerLane;
constexpr uint32_t kStageBytes = kCodesPerStep * 8 + 32;
constexpr uint32_t kMaxInlined = 12;

__device__ __forceinline__ uint32_t apply_map(uint32_t m, uint32_t x) { return (m >> x) & 1u; }
__device__ __forceinline__ uint32_t compose(uint32_t g, uint32_t f) {
    return apply_map(g, f & 1u) | (apply_map(g, (f >> 1) & 1u) << 1);
}

// Decodes `n` codes at `p` into exactly the bytes [out_start, out_start + out_len)
// of `out` with one warp.
__device__ __forceinline__ void decode_group(const uint8_t *__restrict p, uint64_t n, const uint64_t *sym,
                                             const uint8_t *len, uint8_t *__restrict out, uint64_t out_start,
                                             uint64_t out_len, uint8_t *__restrict stage) {
    const uint32_t lane = threadIdx.x % kWarp;
    uint32_t skip = static_cast<uint32_t>(out_start & 15);
    uint8_t *dst = out + (out_start - skip);
    uint32_t fill = skip;
    uint64_t written = 0;
    uint64_t produced = 0;
    uint32_t carry = 0;
    for (uint64_t base = 0; base < n; base += kCodesPerStep) {
        const uint64_t idx = base + lane * kCodesPerLane;
        const uint32_t valid = idx < n ? static_cast<uint32_t>(min(uint64_t{kCodesPerLane}, n - idx)) : 0;
        uint32_t c[kCodesPerLane];
#pragma unroll
        for (uint32_t k = 0; k < kCodesPerLane; ++k) c[k] = k < valid ? __ldg(p + idx + k) : 0u;
        uint32_t map = 0;
#pragma unroll
        for (uint32_t x = 0; x < 2; ++x) {
            uint32_t lit = x;
#pragma unroll
            for (uint32_t k = 0; k < kCodesPerLane; ++k) {
                if (k < valid) lit = lit ? 0u : (c[k] == 255u ? 1u : 0u);
            }
            map |= lit << x;
        }
#pragma unroll
        for (uint32_t d = 1; d < kWarp; d <<= 1) {
            const uint32_t other = __shfl_up_sync(0xffffffffu, map, d);
            if (lane >= d) map = compose(map, other);
        }
        const uint32_t prev = __shfl_up_sync(0xffffffffu, map, 1);
        uint32_t lit = lane == 0 ? carry : apply_map(prev, carry);
        carry = apply_map(__shfl_sync(0xffffffffu, map, kWarp - 1), carry);
        uint64_t bytes[kCodesPerLane];
        uint32_t lens[kCodesPerLane];
        uint32_t total = 0;
#pragma unroll
        for (uint32_t k = 0; k < kCodesPerLane; ++k) {
            uint32_t l = 0;
            uint64_t b = 0;
            if (k < valid) {
                if (lit) {
                    l = 1;
                    b = c[k];
                    lit = 0;
                } else if (c[k] == 255u) {
                    lit = 1;
                } else {
                    l = len[c[k]];
                    b = sym[c[k]];
                }
            }
            lens[k] = l;
            bytes[k] = b;
            total += l;
        }
        uint32_t incl = total;
#pragma unroll
        for (uint32_t d = 1; d < kWarp; d <<= 1) {
            const uint32_t v = __shfl_up_sync(0xffffffffu, incl, d);
            if (lane >= d) incl += v;
        }
        uint32_t step_total = __shfl_sync(0xffffffffu, incl, kWarp - 1);
        // Never write past the group (only possible with invalid codes).
        const uint64_t room = out_len - produced;
        uint32_t pos = fill + incl - total;
        const uint32_t limit = fill + static_cast<uint32_t>(min(uint64_t{step_total}, room));
#pragma unroll
        for (uint32_t k = 0; k < kCodesPerLane; ++k) {
            for (uint32_t b = 0; b < lens[k]; ++b, ++pos) {
                if (pos < limit) stage[pos] = static_cast<uint8_t>(bytes[k] >> (8 * b));
            }
        }
        __syncwarp();
        step_total = limit - fill;
        fill = limit;
        produced += step_total;
        const uint32_t chunks = fill / 16;
        uint32_t j0 = 0;
        if (skip && chunks) {
            if (lane >= skip && lane < 16) dst[written + lane] = stage[lane];
            j0 = 1;
            skip = 0;
        }
        uint4 *gdst = reinterpret_cast<uint4 *>(dst + written);
        const uint4 *sstage = reinterpret_cast<const uint4 *>(stage);
        for (uint32_t j = j0 + lane; j < chunks; j += kWarp) gdst[j] = sstage[j];
        const uint32_t rem = fill - chunks * 16;
        const uint8_t keep = lane < rem ? stage[chunks * 16 + lane] : 0;
        __syncwarp();
        if (lane < rem) stage[lane] = keep;
        __syncwarp();
        written += chunks * 16;
        fill = rem;
        if (produced == out_len && base + kCodesPerStep < n) break;  // invalid codes: stop
    }
    if (lane < fill && lane >= skip) dst[written + lane] = stage[lane];
}

// One string, one lane, byte by byte (fallback for unusual groups).
__device__ __forceinline__ void decode_string_simple(const uint8_t *__restrict codes, uint64_t in_pos, uint64_t in_end,
                                                     const uint64_t *sym, const uint8_t *len,
                                                     uint8_t *__restrict out, uint64_t out_pos, uint64_t out_end) {
    while (in_pos < in_end && out_pos < out_end) {
        const uint32_t c = codes[in_pos++];
        if (c == 255u) {
            if (in_pos < in_end) out[out_pos++] = codes[in_pos++];
        } else {
            const uint64_t s = sym[c];
            for (uint32_t b = 0; b < len[c] && out_pos < out_end; ++b) out[out_pos++] = static_cast<uint8_t>(s >> (8 * b));
        }
    }
}

template <typename CodeOffsetT>
__device__ __forceinline__ void fsst_warp(const uint8_t *__restrict codes_bytes, const CodeOffsetT *__restrict codes_offsets,
                                          const uint64_t *__restrict symbols, const uint8_t *__restrict symbol_lengths,
                                          uint32_t num_symbols, const uint64_t *__restrict output_offsets,
                                          const uint8_t *__restrict validity_bits, uint64_t validity_bit_offset,
                                          uint8_t *__restrict output_bytes, uint4 *__restrict output_views,
                                          uint64_t num_strings) {
    __shared__ uint64_t sym[256];
    __shared__ uint8_t len[256];
    __shared__ __align__(16) uint8_t stage[kWarpsPerBlock][kStageBytes];
    for (uint32_t i = threadIdx.x; i < 256; i += blockDim.x) {
        const bool used = i < num_symbols && i < 255;
        const uint32_t l = used ? symbol_lengths[i] : 0u;
        // Mask the symbol to its length (the high bytes are unspecified).
        sym[i] = used ? (l >= 8 ? symbols[i] : symbols[i] & ((uint64_t{1} << (8 * l)) - 1)) : 0;
        len[i] = static_cast<uint8_t>(l);
    }
    __syncthreads();
    const uint32_t warp = threadIdx.x / kWarp;
    const uint32_t lane = threadIdx.x % kWarp;
    const uint64_t groups = (num_strings + kWarp - 1) / kWarp;
    for (uint64_t g = uint64_t{blockIdx.x} * kWarpsPerBlock + warp; g < groups;
         g += uint64_t{gridDim.x} * kWarpsPerBlock) {
        const uint64_t first = g * kWarp;
        const uint64_t last = min(first + kWarp, num_strings);
        const uint64_t sid = first + lane;
        const bool in_group = sid < last;
        bool is_valid = false;
        bool odd_null = false;
        if (in_group) {
            const uint64_t vi = sid + validity_bit_offset;
            is_valid = ((validity_bits[vi >> 3] >> (vi & 7u)) & 1u) != 0u;
            odd_null = !is_valid && (codes_offsets[sid + 1] != codes_offsets[sid] ||
                                     output_offsets[sid + 1] != output_offsets[sid]);
        }
        if (__any_sync(0xffffffffu, odd_null)) {
            if (in_group && is_valid) {
                decode_string_simple(codes_bytes, codes_offsets[sid], codes_offsets[sid + 1], sym, len, output_bytes,
                                     output_offsets[sid], output_offsets[sid + 1]);
            }
        } else {
            const uint64_t in_start = codes_offsets[first];
            const uint64_t out_start = output_offsets[first];
            decode_group(codes_bytes + in_start, uint64_t(codes_offsets[last]) - in_start, sym, len, output_bytes,
                         out_start, output_offsets[last] - out_start, stage[warp]);
        }
        __syncwarp();
        __threadfence_block();
        if (in_group && output_views != nullptr) {
            if (!is_valid) {
                output_views[sid] = make_uint4(0, 0, 0, 0);
            } else {
                const uint64_t start = output_offsets[sid];
                const uint32_t n = static_cast<uint32_t>(output_offsets[sid + 1] - start);
                if (n <= kMaxInlined) {
                    uint32_t words[3] = {0, 0, 0};
                    for (uint32_t i = 0; i < n; i++) words[i >> 2] |= uint32_t{output_bytes[start + i]} << (8u * (i & 3u));
                    output_views[sid] = make_uint4(n, words[0], words[1], words[2]);
                } else {
                    const uint32_t prefix = uint32_t{output_bytes[start]} | (uint32_t{output_bytes[start + 1]} << 8u) |
                                            (uint32_t{output_bytes[start + 2]} << 16u) |
                                            (uint32_t{output_bytes[start + 3]} << 24u);
                    output_views[sid] = make_uint4(n, prefix, 0, static_cast<uint32_t>(start));
                }
            }
        }
    }
}

}  // namespace

#define GENERATE_FSST_WARP_KERNEL(suffix, CodeOffsetT)                                                            \
    extern "C" __global__ void __launch_bounds__(256)                                                             \
        fsst_warp_##suffix(const uint8_t *__restrict codes_bytes, const CodeOffsetT *__restrict codes_offsets,    \
                           const uint64_t *__restrict symbols, const uint8_t *__restrict symbol_lengths,          \
                           uint32_t num_symbols, const uint64_t *__restrict output_offsets,                       \
                           const uint8_t *__restrict validity_bits, uint64_t validity_bit_offset,                 \
                           uint8_t *__restrict output_bytes, uint4 *__restrict output_views, uint64_t num_strings) { \
        fsst_warp<CodeOffsetT>(codes_bytes, codes_offsets, symbols, symbol_lengths, num_symbols, output_offsets,  \
                               validity_bits, validity_bit_offset, output_bytes, output_views, num_strings);      \
    }

GENERATE_FSST_WARP_KERNEL(u8, uint8_t)
GENERATE_FSST_WARP_KERNEL(u16, uint16_t)
GENERATE_FSST_WARP_KERNEL(u32, uint32_t)
GENERATE_FSST_WARP_KERNEL(u64, uint64_t)
