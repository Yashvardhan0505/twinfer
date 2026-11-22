// Fake CUDA runtime that runs kernels on the CPU, so kernel logic can be tested without a GPU.
// Every CUDA thread is an OS thread and blocks run one at a time. __syncthreads and warp
// shuffles use real barriers, so a divergent sync or partial-warp shuffle deadlocks here
// (which is a real bug on a GPU too). __shared__ becomes a function-local static.
// Launches are rewritten by preprocess.py. See README.md in this folder.
#pragma once
#include <algorithm>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

#define __host__
#define __device__
#define __global__
#define __shared__ static
#define __constant__ static
#define __forceinline__ inline __attribute__((always_inline))
#define __noinline__ __attribute__((noinline))
#define __launch_bounds__(...)
#define __align__(n) __attribute__((aligned(n)))
#define CUDART_VERSION 12000
#define TWINFER_CUDA_EMULATOR 1

struct dim3 {
  unsigned x, y, z;
  constexpr dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};
struct __attribute__((aligned(16))) uint4 { unsigned x, y, z, w; };
struct __attribute__((aligned(8))) uint2 { unsigned x, y; };
struct __attribute__((aligned(16))) int4 { int x, y, z, w; };
struct __attribute__((aligned(8))) int2 { int x, y; };
struct __attribute__((aligned(8))) float2 { float x, y; };
struct __attribute__((aligned(16))) float4 { float x, y, z, w; };
inline uint4 make_uint4(unsigned a, unsigned b, unsigned c, unsigned d) { return {a, b, c, d}; }
inline uint2 make_uint2(unsigned a, unsigned b) { return {a, b}; }
inline float2 make_float2(float a, float b) { return {a, b}; }
inline float4 make_float4(float a, float b, float c, float d) { return {a, b, c, d}; }

namespace emu {

struct WarpCtx {
  std::unique_ptr<std::barrier<>> bar;
  uint64_t buf[32];
  float scratch[256];
};

inline thread_local int tl_lane = 0;
inline thread_local WarpCtx* tl_warp = nullptr;
inline thread_local std::barrier<>* tl_block_bar = nullptr;
inline unsigned char* g_dyn_smem = nullptr;

inline unsigned char* dyn_smem() { return g_dyn_smem; }
inline int lane() { return tl_lane; }
inline void warp_barrier() { tl_warp->bar->arrive_and_wait(); }
inline float* warp_scratch() { return tl_warp->scratch; }

struct Capture {
  bool active = false;
  std::vector<std::function<void()>> ops;
};
inline Capture g_capture;

}  // namespace emu

inline thread_local dim3 threadIdx;
inline thread_local dim3 blockIdx;
inline dim3 blockDim;
inline dim3 gridDim;
constexpr int warpSize = 32;

namespace emu {

template <typename F, typename... A>
void run_kernel(F fn, dim3 grid, dim3 block, size_t smem, A... args) {
  const unsigned nthreads = block.x * block.y * block.z;
  if (nthreads == 0 || nthreads > 1024) {
    std::fprintf(stderr, "emu: bad block size %u\n", nthreads);
    std::abort();
  }
  blockDim = block;
  gridDim = grid;
  std::vector<unsigned char> dyn(smem + 256);
  g_dyn_smem = reinterpret_cast<unsigned char*>((reinterpret_cast<uintptr_t>(dyn.data()) + 127) & ~uintptr_t(127));
  const unsigned nwarps = (nthreads + 31) / 32;
  std::vector<WarpCtx> warps(nwarps);
  for (unsigned w = 0; w < nwarps; ++w)
    warps[w].bar = std::make_unique<std::barrier<>>(std::min<unsigned>(32, nthreads - 32 * w));
  std::barrier<> block_bar(nthreads);
  std::vector<std::thread> threads;
  threads.reserve(nthreads);
  for (unsigned t = 0; t < nthreads; ++t) {
    threads.emplace_back([&, t]() {
      threadIdx = dim3(t % block.x, (t / block.x) % block.y, t / (block.x * block.y));
      tl_lane = static_cast<int>(t % 32);
      tl_warp = &warps[t / 32];
      tl_block_bar = &block_bar;
      for (unsigned bz = 0; bz < grid.z; ++bz)
        for (unsigned by = 0; by < grid.y; ++by)
          for (unsigned bx = 0; bx < grid.x; ++bx) {
            blockIdx = dim3(bx, by, bz);
            fn(args...);
            block_bar.arrive_and_wait();  // whole block done before the next one reuses smem
          }
    });
  }
  for (auto& th : threads) th.join();
}

template <typename F>
struct Launcher {
  F fn;
  dim3 grid, block;
  size_t smem;
  template <typename... A>
  void operator()(A... args) const {
    F f = fn;
    dim3 g = grid, b = block;
    size_t s = smem;
    if (g_capture.active) {
      g_capture.ops.push_back([=]() { run_kernel(f, g, b, s, args...); });
      return;
    }
    run_kernel(f, g, b, s, args...);
  }
};

template <typename F, typename S = void*>
Launcher<F> launch(F fn, dim3 grid, dim3 block, size_t smem = 0, S = nullptr) {
  return Launcher<F>{fn, grid, block, smem};
}

template <typename T>
T shfl(T v, int src) {
  static_assert(sizeof(T) <= 8, "shuffle of >8-byte type");
  uint64_t bits = 0;
  std::memcpy(&bits, &v, sizeof(T));
  tl_warp->buf[tl_lane] = bits;
  warp_barrier();
  const uint64_t r = tl_warp->buf[src];
  warp_barrier();
  T out;
  std::memcpy(&out, &r, sizeof(T));
  return out;
}

}  // namespace emu

// synchronisation & warp intrinsics
inline void __syncthreads() { emu::tl_block_bar->arrive_and_wait(); }
inline void __syncwarp(unsigned = 0xffffffffu) { emu::warp_barrier(); }

template <typename T>
T __shfl_xor_sync(unsigned, T v, int lane_mask, int width = 32) {
  const int lane = emu::tl_lane;
  int src = lane ^ lane_mask;
  if (src / width != lane / width) src = lane;
  return emu::shfl(v, src);
}
template <typename T>
T __shfl_down_sync(unsigned, T v, unsigned delta, int width = 32) {
  const int lane = emu::tl_lane, seg = lane / width * width;
  int src = lane + static_cast<int>(delta);
  if (src >= seg + width) src = lane;
  return emu::shfl(v, src);
}
template <typename T>
T __shfl_sync(unsigned, T v, int src_lane, int width = 32) {
  const int seg = emu::tl_lane / width * width;
  return emu::shfl(v, seg + (src_lane % width));
}

// math & bit intrinsics
inline float rsqrtf(float x) { return 1.0f / std::sqrt(x); }
inline float __expf(float x) { return std::exp(x); }
inline float __fdividef(float a, float b) { return a / b; }
inline float __int_as_float(int i) { float f; std::memcpy(&f, &i, 4); return f; }
inline float __int_as_float(unsigned i) { float f; std::memcpy(&f, &i, 4); return f; }
inline float __uint_as_float(unsigned i) { float f; std::memcpy(&f, &i, 4); return f; }
inline int __float_as_int(float f) { int i; std::memcpy(&i, &f, 4); return i; }
inline unsigned __float_as_uint(float f) { unsigned i; std::memcpy(&i, &f, 4); return i; }
inline unsigned __byte_perm(unsigned x, unsigned y, unsigned s) {
  const uint64_t v = (static_cast<uint64_t>(y) << 32) | x;
  unsigned r = 0;
  for (int i = 0; i < 4; ++i) {
    const unsigned sel = (s >> (4 * i)) & 7u;
    r |= static_cast<unsigned>((v >> (8 * sel)) & 0xffu) << (8 * i);
  }
  return r;
}
template <typename T>
inline T __ldg(const T* p) { return *p; }
inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }
inline unsigned min(unsigned a, unsigned b) { return a < b ? a : b; }
inline unsigned max(unsigned a, unsigned b) { return a > b ? a : b; }
inline long long min(long long a, long long b) { return a < b ? a : b; }
inline long long max(long long a, long long b) { return a > b ? a : b; }
#include <math.h>  // expf, exp2f, fmaf, fmaxf, sqrtf at global scope, as in device code

// runtime API
enum cudaError_t { cudaSuccess = 0, cudaErrorMemoryAllocation = 2, cudaErrorInvalidValue = 1 };
enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4
};
enum cudaStreamCaptureMode {
  cudaStreamCaptureModeGlobal = 0,
  cudaStreamCaptureModeThreadLocal = 1,
  cudaStreamCaptureModeRelaxed = 2
};
struct CUstream_st;
typedef CUstream_st* cudaStream_t;
struct EmuEvent { double t = 0; };
typedef EmuEvent* cudaEvent_t;
struct EmuGraph { std::vector<std::function<void()>> ops; };
typedef EmuGraph* cudaGraph_t;
typedef EmuGraph* cudaGraphExec_t;

struct cudaDeviceProp {
  char name[256];
  int major, minor, multiProcessorCount;
  size_t totalGlobalMem;
  int l2CacheSize;
};
enum cudaDeviceAttr { cudaDevAttrMemoryClockRate = 36, cudaDevAttrGlobalMemoryBusWidth = 37 };
inline cudaError_t cudaDeviceGetAttribute(int* v, cudaDeviceAttr a, int) {
  *v = a == cudaDevAttrMemoryClockRate ? 1000000 : 64;  // pretend: 1 GHz, 64-bit bus
  return cudaSuccess;
}

inline const char* cudaGetErrorString(cudaError_t) { return "emulator error"; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaPeekAtLastError() { return cudaSuccess; }
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) {
  std::memset(p, 0, sizeof(*p));
  std::snprintf(p->name, sizeof(p->name), "CPU emulator");
  p->major = 8;
  p->minor = 0;
  p->multiProcessorCount = 1;
  p->totalGlobalMem = size_t(1) << 32;
  p->l2CacheSize = 1 << 20;
  return cudaSuccess;
}
inline cudaError_t cudaMalloc(void** p, size_t n) {
  *p = std::aligned_alloc(256, (n + 255) / 256 * 256);
  return *p ? cudaSuccess : cudaErrorMemoryAllocation;
}
template <typename T>
inline cudaError_t cudaMalloc(T** p, size_t n) { return cudaMalloc(reinterpret_cast<void**>(p), n); }
inline cudaError_t cudaMallocHost(void** p, size_t n) { return cudaMalloc(p, n); }
template <typename T>
inline cudaError_t cudaMallocHost(T** p, size_t n) { return cudaMalloc(reinterpret_cast<void**>(p), n); }
inline cudaError_t cudaFree(void* p) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaFreeHost(void* p) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) {
  std::memmove(d, s, n);
  return cudaSuccess;
}
inline cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind k, cudaStream_t = nullptr) {
  if (emu::g_capture.active) { std::fprintf(stderr, "emu: memcpy during capture not supported\n"); std::abort(); }
  return cudaMemcpy(d, s, n, k);
}
inline cudaError_t cudaMemset(void* d, int v, size_t n) { std::memset(d, v, n); return cudaSuccess; }
inline cudaError_t cudaMemsetAsync(void* d, int v, size_t n, cudaStream_t = nullptr) { return cudaMemset(d, v, n); }
inline cudaError_t cudaStreamCreate(cudaStream_t* s) { *s = reinterpret_cast<cudaStream_t>(new int(0)); return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t s) { delete reinterpret_cast<int*>(s); return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline double emu_now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline cudaError_t cudaEventCreate(cudaEvent_t* e) { *e = new EmuEvent; return cudaSuccess; }
inline cudaError_t cudaEventDestroy(cudaEvent_t e) { delete e; return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t = nullptr) { e->t = emu_now_ms(); return cudaSuccess; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b) {
  *ms = static_cast<float>(b->t - a->t);
  return cudaSuccess;
}
inline cudaError_t cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode) {
  emu::g_capture.active = true;
  emu::g_capture.ops.clear();
  return cudaSuccess;
}
inline cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t* g) {
  *g = new EmuGraph{std::move(emu::g_capture.ops)};
  emu::g_capture.active = false;
  emu::g_capture.ops.clear();
  return cudaSuccess;
}
inline cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t* e, cudaGraph_t g, unsigned long long) {
  *e = new EmuGraph{g->ops};
  return cudaSuccess;
}
inline cudaError_t cudaGraphLaunch(cudaGraphExec_t e, cudaStream_t) {
  for (auto& op : e->ops) op();
  return cudaSuccess;
}
inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t e) { delete e; return cudaSuccess; }
inline cudaError_t cudaGraphDestroy(cudaGraph_t g) { delete g; return cudaSuccess; }
