// presst GPU encoding (U8 scheme): greedy longest match, byte-identical to the
// CPU encoder. Header-only device code.
//
// Matching at each position, as in presst's normative encoding: the dictionary
// has at most one symbol of 3+ bytes per 3-byte prefix, so the longest match is
// that symbol if it matches completely, else the 2-byte symbol of the next two
// bytes, else the 1-byte symbol, else an escape (255) plus the literal byte.
#pragma once

#include <stdint.h>

#include "presst_decode.cuh"

namespace presst {

constexpr uint32_t kSlotBits = 10;
constexpr uint32_t kLongSlots = 1u << kSlotBits;  // linear probing
constexpr uint32_t kTwoSlots = 1u << kSlotBits;
constexpr uint32_t kHashMul = 0x9E3779B1u;

// 14592 bytes; built on the host from the dictionary, uploaded once.
struct EncodeTable {
    uint64_t long_sym[kLongSlots];  // symbol bytes (zero padded), 0 if empty
    uint8_t long_len[kLongSlots];   // 3..8, 0 for an empty slot
    uint8_t long_code[kLongSlots];
    uint32_t two[kTwoSlots];        // key (16 bits) | code << 16 | 1 << 24; 0 if empty
    uint8_t byte_code[256];         // code of the 1-byte symbol, 255 if none
};

// Must match `hash_slot` in src/encode.rs.
__device__ __forceinline__ uint32_t hash_slot(uint32_t key) { return (key * kHashMul) >> (32 - kSlotBits); }

// Input window: the 8 bytes at `pos` (the input buffer has 16 bytes of slack).
__device__ __forceinline__ uint64_t load_window(const uint8_t* __restrict__ in, uint64_t pos) {
    const uint64_t* words = reinterpret_cast<const uint64_t*>(in + (pos & ~uint64_t{7}));
    const uint32_t shift = static_cast<uint32_t>(pos & 7) * 8;
    const uint64_t lo = __ldg(words);
    if (shift == 0) return lo;
    const uint64_t hi = __ldg(words + 1);
    return (lo >> shift) | (hi << (64 - shift));
}

// One unit of encode work: input bytes [in_start, in_start + in_len), codes to
// out[out_start ..] (a multiple of 8, room for 2 * in_len bytes plus 8 slack).
struct Range {
    uint64_t in_start;
    uint64_t out_start;
    uint32_t in_len;
    uint32_t _pad;
};

// Encode tables as seen by a kernel (shared memory).
struct EncodeTables {
    const uint64_t* long_sym;
    const uint8_t* long_len;
    const uint8_t* long_code;
    const uint32_t* two;
    const uint8_t* byte_code;
};

// The greedy longest match at a position: `win` holds the next 8 input bytes
// (bytes past the end are ignored), `remaining` how many are left (> 0).
// Sets `out`/`nout` to the code bytes to emit (1, or 2 for escape + literal)
// and returns how many input bytes they cover.
//
// All three lookups always run and the result is selected without branches, so
// the lanes of a warp stay on one path (they would otherwise take different
// branches at almost every code). Only hash collisions loop, which is rare: the
// tables are at most a quarter full.
__device__ __forceinline__ uint32_t match_at(uint64_t win, uint64_t remaining, const EncodeTables& t,
                                             uint64_t& out, uint32_t& nout) {
    // Long symbol (3..8 bytes) with these 3 bytes as prefix, if any.
    const uint32_t key3 = static_cast<uint32_t>(win & 0xFFFFFFu);
    uint32_t slot3 = hash_slot(key3);
    uint32_t l3 = t.long_len[slot3];
    uint64_t sym3 = t.long_sym[slot3];
    while (l3 != 0 && static_cast<uint32_t>(sym3 & 0xFFFFFFu) != key3) {
        slot3 = (slot3 + 1) & (kLongSlots - 1);
        l3 = t.long_len[slot3];
        sym3 = t.long_sym[slot3];
    }
    const uint64_t mask3 = l3 >= 8 ? ~uint64_t{0} : ((uint64_t{1} << (8 * l3)) - 1);
    const bool long_hit = l3 != 0 && l3 <= remaining && (win & mask3) == sym3;
    // 2-byte symbol of the next two bytes, if any.
    const uint32_t key2 = static_cast<uint32_t>(win & 0xFFFFu);
    uint32_t slot2 = hash_slot(key2);
    uint32_t e2 = t.two[slot2];
    while (e2 != 0 && (e2 & 0xFFFFu) != key2) {
        slot2 = (slot2 + 1) & (kTwoSlots - 1);
        e2 = t.two[slot2];
    }
    const bool two_hit = e2 != 0 && remaining >= 2;
    // 1-byte symbol, else escape.
    const uint32_t b = static_cast<uint32_t>(win & 0xFFu);
    const uint32_t c1 = t.byte_code[b];
    const bool one_hit = c1 != 255u;

    const uint32_t code = long_hit ? t.long_code[slot3] : two_hit ? ((e2 >> 16) & 0xFFu) : c1;
    const bool escape = !long_hit && !two_hit && !one_hit;
    out = escape ? (0xFFu | (static_cast<uint64_t>(b) << 8)) : code;
    nout = escape ? 2u : 1u;
    return long_hit ? l3 : two_hit ? 2u : 1u;
}

// Encodes one range, reading the input from global memory; returns the number
// of code bytes written.
__device__ __forceinline__ uint32_t encode_range(const uint8_t* __restrict__ in, const Range& r,
                                                 const EncodeTables& t, uint8_t* __restrict__ out) {
    AlignedWriter w{reinterpret_cast<uint64_t*>(out + r.out_start), 0, 0};
    uint32_t written = 0;
    uint64_t pos = r.in_start;
    const uint64_t end = r.in_start + r.in_len;
    while (pos < end) {
        uint64_t code;
        uint32_t n;
        pos += match_at(load_window(in, pos), end - pos, t, code, n);
        w.put(code, n);
        written += n;
    }
    w.finish();
    return written;
}

// Tiled encoding: one warp encodes 32 consecutive ranges (lane i: range i), in
// steps of kTile input bytes per lane. Each step, the warp loads every lane's
// next kTile + 16 bytes into a shared-memory row with coalesced 4-byte loads,
// then every lane matches from its own row. Range starts must be multiples of 4,
// and the input needs kTile + 16 bytes of slack.
constexpr uint32_t kTile = 64;
constexpr uint32_t kRowBytes = kTile + 24;  // 88: room for the look-ahead, 8-byte aligned

__device__ __forceinline__ uint64_t row_window(const uint8_t* row, uint32_t off) {
    const uint64_t* words = reinterpret_cast<const uint64_t*>(row + (off & ~7u));
    const uint32_t shift = (off & 7u) * 8;
    const uint64_t lo = words[0];
    if (shift == 0) return lo;
    return (lo >> shift) | (words[1] << (64 - shift));
}

// Output is interleaved: word j of the lane goes to out_start + j * 256 (the
// lanes of a warp have consecutive 8-byte slots), so the stores of a warp land
// next to each other.
__device__ __forceinline__ uint32_t encode_range_tiled(const uint8_t* __restrict__ in, const Range& r,
                                                       const EncodeTables& t, uint8_t* __restrict__ out,
                                                       uint8_t* __restrict__ rows /* 32 * kRowBytes */) {
    const uint32_t lane = threadIdx.x & 31u;
    uint32_t max_len = r.in_len;
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) max_len = max(max_len, __shfl_xor_sync(0xffffffffu, max_len, d));
    StridedWriter<32> w{reinterpret_cast<uint64_t*>(out + r.out_start), 0, 0};
    uint32_t written = 0;
    uint32_t pos = 0;  // relative to the range start
    uint8_t* my_row = rows + lane * kRowBytes;
    for (uint32_t base = 0; base < max_len; base += kTile) {
        // Load kTile + 16 bytes of every lane's range, one lane's row at a time.
        for (uint32_t src = 0; src < 32; ++src) {
            const uint64_t start = __shfl_sync(0xffffffffu, r.in_start, src);
            if (lane < (kTile + 16) / 4) {
                const uint32_t v = __ldg(reinterpret_cast<const uint32_t*>(in + start + base) + lane);
                reinterpret_cast<uint32_t*>(rows + src * kRowBytes)[lane] = v;
            }
        }
        __syncwarp();
        // A match may run up to 7 bytes past the tile; the next tile starts there.
        const uint32_t tile_end = min(base + kTile, r.in_len);
        while (pos < tile_end) {
            uint64_t code;
            uint32_t n;
            pos += match_at(row_window(my_row, pos - base), r.in_len - pos, t, code, n);
            w.put(code, n);
            written += n;
        }
        __syncwarp();
    }
    w.finish();
    return written;
}

}  // namespace presst
