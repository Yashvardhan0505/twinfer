// CPU emulation of the cuda_fp16.h subset used by twinfer (see cuda_runtime.h).
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

#include "cuda_runtime.h"

namespace emu {
inline float h2f(uint16_t h) {
  const uint32_t sign = (h & 0x8000u) << 16, exp = (h >> 10) & 0x1Fu, mant = h & 0x3FFu;
  uint32_t u;
  if (exp == 0) {
    float v = std::ldexp(static_cast<float>(mant), -24);
    return sign ? -v : v;
  }
  if (exp == 31) u = sign | 0x7F800000u | (mant << 13);
  else u = sign | ((exp + 112u) << 23) | (mant << 13);
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
inline uint16_t f2h(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u, ax = x & 0x7FFFFFFFu;
  if (ax >= 0x7F800000u) return static_cast<uint16_t>(sign | 0x7C00u | (ax > 0x7F800000u ? 0x200u : 0u));
  if (ax >= 0x477FF000u) return static_cast<uint16_t>(sign | 0x7C00u);
  if (ax < 0x38800000u) {
    float a;
    std::memcpy(&a, &ax, 4);
    return static_cast<uint16_t>(sign | static_cast<uint32_t>(std::nearbyint(a * 16777216.0f)));
  }
  uint32_t h = (((ax >> 23) - 112u) << 10) | ((ax & 0x7FFFFFu) >> 13);
  const uint32_t rem = ax & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
  return static_cast<uint16_t>(sign | h);
}
}  // namespace emu

struct __half {
  uint16_t x;
};
struct __half2 {
  __half x, y;
};
typedef __half half;
typedef __half2 half2;

inline __half __float2half(float f) { return __half{emu::f2h(f)}; }
inline __half __float2half_rn(float f) { return __float2half(f); }
inline float __half2float(__half h) { return emu::h2f(h.x); }
inline float __low2float(__half2 h) { return __half2float(h.x); }
inline float __high2float(__half2 h) { return __half2float(h.y); }
inline float2 __half22float2(__half2 h) { return float2{__half2float(h.x), __half2float(h.y)}; }
inline __half2 __float22half2_rn(float2 f) { return __half2{__float2half(f.x), __float2half(f.y)}; }
inline __half2 __float2half2_rn(float f) { return __half2{__float2half(f), __float2half(f)}; }
inline __half2 __floats2half2_rn(float a, float b) { return __half2{__float2half(a), __float2half(b)}; }
inline __half2 __halves2half2(__half a, __half b) { return __half2{a, b}; }
inline __half2 __hsub2(__half2 a, __half2 b) {
  return __half2{__float2half(__half2float(a.x) - __half2float(b.x)), __float2half(__half2float(a.y) - __half2float(b.y))};
}
inline __half2 __hadd2(__half2 a, __half2 b) {
  return __half2{__float2half(__half2float(a.x) + __half2float(b.x)), __float2half(__half2float(a.y) + __half2float(b.y))};
}
inline __half2 __hmul2(__half2 a, __half2 b) {
  return __half2{__float2half(__half2float(a.x) * __half2float(b.x)), __float2half(__half2float(a.y) * __half2float(b.y))};
}
