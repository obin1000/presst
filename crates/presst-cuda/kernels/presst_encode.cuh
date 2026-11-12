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

// Encodes one range; returns the number of code bytes written.
__device__ __forceinline__ uint32_t encode_range(const uint8_t* __restrict__ in,
                                                 const Range& r,
                                                 const uint64_t* __restrict__ long_sym,
                                                 const uint8_t* __restrict__ long_len,
                                                 const uint8_t* __restrict__ long_code,
                                                 const uint32_t* __restrict__ two,
                                                 const uint8_t* __restrict__ byte_code,
                                                 uint8_t* __restrict__ out) {
    AlignedWriter w{reinterpret_cast<uint64_t*>(out + r.out_start), 0, 0};
    uint32_t written = 0;
    uint64_t pos = r.in_start;
    const uint64_t end = r.in_start + r.in_len;
    while (pos < end) {
        const uint64_t win = load_window(in, pos);
        const uint64_t remaining = end - pos;
        uint32_t code = 0;
        uint32_t mlen = 0;
        if (remaining >= 3) {
            const uint32_t key = static_cast<uint32_t>(win & 0xFFFFFFu);
            for (uint32_t slot = hash_slot(key);; slot = (slot + 1) & (kLongSlots - 1)) {
                const uint32_t l = long_len[slot];
                if (l == 0) break;
                const uint64_t sym = long_sym[slot];
                if (static_cast<uint32_t>(sym & 0xFFFFFFu) == key) {
                    const uint64_t mask = l == 8 ? ~uint64_t{0} : ((uint64_t{1} << (8 * l)) - 1);
                    if (l <= remaining && (win & mask) == sym) {
                        code = long_code[slot];
                        mlen = l;
                    }
                    break;
                }
            }
        }
        if (mlen == 0 && remaining >= 2) {
            const uint32_t key = static_cast<uint32_t>(win & 0xFFFFu);
            for (uint32_t slot = hash_slot(key);; slot = (slot + 1) & (kTwoSlots - 1)) {
                const uint32_t e = two[slot];
                if (e == 0) break;
                if ((e & 0xFFFFu) == key) {
                    code = (e >> 16) & 0xFFu;
                    mlen = 2;
                    break;
                }
            }
        }
        if (mlen == 0) {
            const uint32_t b = static_cast<uint32_t>(win & 0xFFu);
            const uint32_t c = byte_code[b];
            if (c != 255u) {
                code = c;
                mlen = 1;
            } else {
                // Escape + literal.
                w.put(0xFFu | (static_cast<uint64_t>(b) << 8), 2);
                written += 2;
                pos += 1;
                continue;
            }
        }
        w.put(code, 1);
        written += 1;
        pos += mlen;
    }
    w.finish();
    return written;
}

}  // namespace presst
