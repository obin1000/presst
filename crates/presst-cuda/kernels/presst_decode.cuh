// presst GPU decoding (U8 scheme, FSST-compatible code streams).
//
// Header-only device code, so other projects (Vortex) can include it.
//
// Code stream: byte c < 255 emits symbol c (1..8 bytes, little-endian in sym[c],
// length len[c]; len 0 marks a code that is not in the dictionary); byte 255 is an
// escape: the next byte is emitted literally.
#pragma once

#include <stdint.h>

namespace presst {

// 2304 bytes; uploaded once per dictionary.
struct DecodeTable {
    uint64_t sym[256];
    uint8_t len[256];
};

// One unit of independent work: decode (or copy, for raw blocks) `in_len` input
// bytes into exactly `out_len` output bytes at `out_start`.
//
// `out_start` must be a multiple of 8, and the output buffer must have 8 bytes of
// slack after its end: the decoder writes whole aligned 8-byte words, and only the
// final split of a frame can end inside a word.
struct Split {
    uint64_t in_start;
    uint64_t out_start;
    uint32_t in_len;
    uint32_t out_len;
    uint32_t raw;  // 1: copy the input as is
    uint32_t _pad;
};

// Error bits, OR-ed into a device word.
enum : uint32_t {
    ERR_INVALID_CODE = 1u << 0,  // a code that is not in the dictionary
    ERR_TRUNCATED = 1u << 1,     // escape as the last code byte
    ERR_OVERFLOW = 1u << 2,      // a split decodes to more than out_len bytes
    ERR_UNDERFLOW = 1u << 3,     // a split decodes to fewer than out_len bytes
};

// Accumulates output bytes into aligned 8-byte stores.
struct AlignedWriter {
    uint64_t* dst;   // next aligned word to store
    uint64_t acc;    // pending bytes (low `fill` bytes valid)
    uint32_t fill;   // 0..7

    __device__ __forceinline__ void put(uint64_t bytes, uint32_t n) {
        // n in 1..=8; bytes above n are zero.
        acc |= bytes << (fill * 8);
        uint32_t total = fill + n;
        if (total >= 8) {
            *dst++ = acc;
            // fill > 0 here unless n == 8 and fill == 0 (then nothing carries over).
            acc = fill ? bytes >> ((8 - fill) * 8) : 0;
            total -= 8;
        }
        fill = total;
    }

    __device__ __forceinline__ void finish() {
        // The final partial word: the slack after the output makes this safe.
        if (fill) {
            *dst = acc;
        }
    }
};

// Decodes one split. Returns error bits (0 on success). Never writes outside
// [out_start, out_start + round_up(out_len, 8)).
template <bool kSharedTable>
__device__ __forceinline__ uint32_t decode_split(const uint8_t* __restrict__ in,
                                                 const Split& s,
                                                 const uint64_t* __restrict__ sym,
                                                 const uint8_t* __restrict__ len,
                                                 uint8_t* __restrict__ out) {
    const uint8_t* p = in + s.in_start;
    const uint8_t* end = p + s.in_len;
    AlignedWriter w{reinterpret_cast<uint64_t*>(out + s.out_start), 0, 0};
    uint32_t produced = 0;
    if (s.raw) {
        for (; p < end; ++p) {
            w.put(*p, 1);
        }
        w.finish();
        return 0;
    }
    while (p < end) {
        uint32_t c = __ldg(p++);
        uint64_t bytes;
        uint32_t n;
        if (c == 255) {
            if (p == end) {
                w.finish();
                return ERR_TRUNCATED;
            }
            bytes = __ldg(p++);
            n = 1;
        } else {
            n = kSharedTable ? len[c] : __ldg(len + c);
            if (n == 0) {
                w.finish();
                return ERR_INVALID_CODE;
            }
            bytes = kSharedTable ? sym[c] : __ldg(sym + c);
        }
        if (produced + n > s.out_len) {
            w.finish();
            return ERR_OVERFLOW;
        }
        w.put(bytes, n);
        produced += n;
    }
    w.finish();
    return produced == s.out_len ? 0 : ERR_UNDERFLOW;
}

// ---------------------------------------------------------------------------
// K2: one warp per split, 4 codes per lane (128 codes per step).
//
// Escapes: a code is a literal byte if the code before it is an escape (255)
// that is not itself a literal. Each lane describes its 4 codes as a function
// "is the next code a literal?" of "is my first code a literal?" (2 bits); a
// warp scan composes these, so every lane learns its own carry-in.
// Positions: a warp prefix sum of the decoded lengths. Bytes are staged in
// shared memory and drained to global memory as aligned 16-byte stores.

constexpr uint32_t kWarp = 32;
constexpr uint32_t kCodesPerLane = 4;
constexpr uint32_t kCodesPerStep = kWarp * kCodesPerLane;
// Worst case per step (128 codes x 8 bytes) plus a carried remainder (< 16).
constexpr uint32_t kStageBytes = kCodesPerStep * 8 + 32;

__device__ __forceinline__ uint32_t warp_or(uint32_t v) {
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) v |= __shfl_xor_sync(0xffffffffu, v, d);
    return v;
}

// Carry map: bit 0 = result when the input is 0, bit 1 = result when it is 1.
__device__ __forceinline__ uint32_t apply_map(uint32_t m, uint32_t x) { return (m >> x) & 1u; }
// "g after f".
__device__ __forceinline__ uint32_t compose(uint32_t g, uint32_t f) {
    return apply_map(g, f & 1u) | (apply_map(g, (f >> 1) & 1u) << 1);
}

template <bool kSharedTable>
__device__ __forceinline__ uint32_t decode_split_warp(const uint8_t* __restrict__ in,
                                                      const Split& s,
                                                      const uint64_t* __restrict__ sym,
                                                      const uint8_t* __restrict__ len,
                                                      uint8_t* __restrict__ out,
                                                      uint8_t* __restrict__ stage) {
    const uint32_t lane = threadIdx.x & (kWarp - 1);
    uint8_t* dst = out + s.out_start;
    if (s.raw) {
        for (uint32_t i = lane; i < s.in_len; i += kWarp) dst[i] = __ldg(in + s.in_start + i);
        return 0;
    }
    const uint8_t* p = in + s.in_start;
    const uint32_t n = s.in_len;
    uint32_t fill = 0;      // staged bytes not yet written (< 16 between steps)
    uint32_t written = 0;   // bytes written to dst (a multiple of 16)
    uint32_t produced = 0;  // bytes decoded so far
    uint32_t carry = 0;     // whether the next step starts with a literal
    uint32_t error = 0;

    for (uint32_t base = 0; base < n; base += kCodesPerStep) {
        const uint32_t idx = base + lane * kCodesPerLane;
        const uint32_t valid = idx < n ? min(kCodesPerLane, n - idx) : 0;
        uint32_t c[kCodesPerLane];
#pragma unroll
        for (uint32_t k = 0; k < kCodesPerLane; ++k) c[k] = k < valid ? __ldg(p + idx + k) : 0u;

        // This lane's carry map over its codes.
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
        // Inclusive scan of maps (lane i: lanes 0..=i composed).
#pragma unroll
        for (uint32_t d = 1; d < kWarp; d <<= 1) {
            uint32_t other = __shfl_up_sync(0xffffffffu, map, d);
            if (lane >= d) map = compose(map, other);
        }
        const uint32_t prev = __shfl_up_sync(0xffffffffu, map, 1);
        uint32_t lit = lane == 0 ? carry : apply_map(prev, carry);
        carry = apply_map(__shfl_sync(0xffffffffu, map, kWarp - 1), carry);

        // Decoded lengths and bytes of this lane's codes.
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
                    l = kSharedTable ? len[c[k]] : __ldg(len + c[k]);
                    if (l == 0) error |= ERR_INVALID_CODE;
                    b = kSharedTable ? sym[c[k]] : __ldg(sym + c[k]);
                }
            }
            lens[k] = l;
            bytes[k] = b;
            total += l;
        }
        if (lit && idx + valid == n && valid > 0) error |= ERR_TRUNCATED;

        // Exclusive prefix sum of the lane totals.
        uint32_t incl = total;
#pragma unroll
        for (uint32_t d = 1; d < kWarp; d <<= 1) {
            uint32_t v = __shfl_up_sync(0xffffffffu, incl, d);
            if (lane >= d) incl += v;
        }
        const uint32_t step_total = __shfl_sync(0xffffffffu, incl, kWarp - 1);
        if (produced + step_total > s.out_len) error |= ERR_OVERFLOW;
        error = warp_or(error);
        if (error) break;

        // Stage this step's bytes.
        uint32_t pos = fill + incl - total;
#pragma unroll
        for (uint32_t k = 0; k < kCodesPerLane; ++k) {
            for (uint32_t b = 0; b < lens[k]; ++b) stage[pos++] = static_cast<uint8_t>(bytes[k] >> (8 * b));
        }
        __syncwarp();
        fill += step_total;
        produced += step_total;

        // Drain whole 16-byte chunks; keep the remainder at the start of the stage.
        const uint32_t chunks = fill / 16;
        uint4* gdst = reinterpret_cast<uint4*>(dst + written);
        const uint4* sstage = reinterpret_cast<const uint4*>(stage);
        for (uint32_t j = lane; j < chunks; j += kWarp) gdst[j] = sstage[j];
        const uint32_t rem = fill - chunks * 16;
        uint8_t keep = lane < rem ? stage[chunks * 16 + lane] : 0;
        __syncwarp();
        if (lane < rem) stage[lane] = keep;
        __syncwarp();
        written += chunks * 16;
        fill = rem;
    }
    if (!error && produced != s.out_len) error = ERR_UNDERFLOW;
    if (!error && lane < fill) dst[written + lane] = stage[lane];
    return error;
}

}  // namespace presst
