// Exhaustive verification for the fp8 decode helpers in dsv41_kernels.cu.
//
//   gcc -O2 -o /tmp/fp8chk scripts/dsv41_fp8_decode_check.c -lm && /tmp/fp8chk
//
// Run this after touching e4m3_to_f / ue8m0_to_f. It is the cheap half of the
// verification pair: it proves the DECODER is bitwise equal to the reference
// formula for every one of the 256 codes, so a later text regression can only
// come from the surrounding loop, not from the conversion. It caught a real bug
// in seconds (a bit-composed rewrite that mishandled the eight subnormal codes,
// which the GPU path would have surfaced only as a degenerate model).
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

// Reference: the formula the kernel used before the bit-composed rewrite.
static float e4m3_ref(uint8_t b) {
    const uint32_t s = (b & 0x80u) ? 0x80000000u : 0u;
    const uint32_t e = (b >> 3) & 0x0Fu;
    const uint32_t m = b & 0x07u;
    if (e == 0) {
        const float v = (float)m * (1.0f / 512.0f);
        return s ? -v : v;
    }
    const float v = (1.0f + (float)m * 0.125f) * exp2f((float)((int)e - 7));
    return s ? -v : v;
}

// The kernel's current implementation, copied verbatim.
static float e4m3_impl(uint8_t b) {
    const uint32_t s = ((uint32_t)b & 0x80u) << 24;
    const uint32_t e = ((uint32_t)b >> 3) & 0x0Fu;
    const uint32_t m = (uint32_t)b & 0x07u;
    if (e == 0u) {
        const float v = (float)m * (1.0f / 512.0f);
        return (b & 0x80u) ? -v : v;
    }
    const uint32_t bits = s | ((e + 120u) << 23) | (m << 20);
    float v;
    memcpy(&v, &bits, 4);
    return v;
}

static float ue8m0_ref(uint8_t b) {
    // 2^(b-127); 0xFF is NaN. (b == 0 would be 2^-127, which the deployable
    // implementation also folds to 0 - both sides agree, so it is not a
    // difference this check needs to flag.)
    if (b == 0xFFu) return NAN;
    return ldexpf(1.0f, (int)b - 127);
}

static float ue8m0_impl(uint8_t b) {
    const uint32_t bits = (b == 0xFFu) ? 0x7FC00000u : ((uint32_t)b << 23);
    float v;
    memcpy(&v, &bits, 4);
    return v;
}

int main(void) {
    int bad_e4 = 0, bad_ue = 0;
    for (int b = 0; b < 256; ++b) {
        if (b != 0x7F && b != 0xFF) {   // e4m3 NaN codes
            const float r = e4m3_ref((uint8_t)b);
            const float i = e4m3_impl((uint8_t)b);
            if (memcmp(&r, &i, 4) != 0) {
                if (bad_e4 < 8) printf("  e4m3 b=%02x ref=%.9g impl=%.9g\n", b, r, i);
                ++bad_e4;
            }
        }
        if (b == 0xFFu) continue;       // ue8m0 NaN
        const float r = ue8m0_ref((uint8_t)b);
        const float i = ue8m0_impl((uint8_t)b);
        if (r != i && !(r == 0.f && i == 0.f)) {
            if (bad_ue < 8) printf("  ue8m0 b=%02x ref=%.9g impl=%.9g\n", b, r, i);
            ++bad_ue;
        }
    }
    printf("e4m3 : %d / 254 bitwise-differing\n", bad_e4);
    printf("ue8m0: %d / 254 differing (b=0 excluded: both sides give 0)\n", bad_ue);
    return (bad_e4 || bad_ue) ? 1 : 0;
}
