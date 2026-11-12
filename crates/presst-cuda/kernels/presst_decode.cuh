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

}  // namespace presst
