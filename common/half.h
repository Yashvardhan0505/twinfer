// fp16 / bf16 <-> fp32 on the host (round to nearest even)
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

namespace tf {

inline float bits_to_f32(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

inline uint32_t f32_to_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}

inline float bf16_to_f32(uint16_t h) { return bits_to_f32(static_cast<uint32_t>(h) << 16); }

inline float f16_to_f32(uint16_t h) {
  const uint32_t sign = (h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t mant = h & 0x3FFu;
  if (exp == 0) {
    float v = std::ldexp(static_cast<float>(mant), -24);  // zero or subnormal
    return sign ? -v : v;
  }
  if (exp == 31) return bits_to_f32(sign | 0x7F800000u | (mant << 13));  // inf / nan
  return bits_to_f32(sign | ((exp + 112u) << 23) | (mant << 13));
}

inline uint16_t f32_to_f16(float f) {
  const uint32_t x = f32_to_bits(f);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const uint32_t ax = x & 0x7FFFFFFFu;
  if (ax >= 0x7F800000u) return static_cast<uint16_t>(sign | 0x7C00u | (ax > 0x7F800000u ? 0x200u : 0u));
  if (ax >= 0x477FF000u) return static_cast<uint16_t>(sign | 0x7C00u);  // rounds to inf
  if (ax < 0x38800000u) {                                                // below 2^-14: subnormal
    const uint32_t m = static_cast<uint32_t>(std::nearbyint(bits_to_f32(ax) * 16777216.0f));
    return static_cast<uint16_t>(sign | m);
  }
  uint32_t h = (((ax >> 23) - 112u) << 10) | ((ax & 0x7FFFFFu) >> 13);
  const uint32_t rem = ax & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
  return static_cast<uint16_t>(sign | h);
}

// for weights: clamp to +-65504 instead of turning into inf
inline uint16_t f32_to_f16_sat(float f) {
  if (f > 65504.0f) f = 65504.0f;
  if (f < -65504.0f) f = -65504.0f;
  return f32_to_f16(f);
}

}  // namespace tf
