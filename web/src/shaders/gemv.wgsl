// Decode GEMV: out = W x. W is row-major [N, K] fp16, packed two per u32 and read as
// vec4<u32> (8 values), unpacked with unpack2x16float so shader-f16 isn't needed.
// 32 invocations per row, tree reduction in workgroup memory (no subgroups).
//
// NORM: fold rmsnorm in, dot(W, x*gamma) * rsqrt(mean(x^2) + eps). Each group reads all of x
//       anyway, so the sum of squares is free.
// MODE: 0 = store (+bias), 1 = add to out (residual), 2 = silu(gate) * up, up rows after the N gate rows

override K: u32;
override N: u32;
override NORM: bool = false;
override MODE: u32 = 0u;
override HAS_BIAS: bool = false;
override EPS: f32 = 1e-5;

@group(0) @binding(0) var<storage, read> W: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> gamma: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

var<workgroup> red_a: array<f32, 256>;
var<workgroup> red_b: array<f32, 256>;
var<workgroup> red_s: array<f32, 256>;

fn dot8(w: vec4<u32>, xa: vec4<f32>, xb: vec4<f32>) -> f32 {
  let lo = vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y));
  let hi = vec4<f32>(unpack2x16float(w.z), unpack2x16float(w.w));
  return dot(lo, xa) + dot(hi, xb);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let lane = tid & 31u;
  let row = (wg.y * nwg.x + wg.x) * 8u + (tid >> 5u);
  let valid = row < N;
  let r = select(0u, row, valid);
  let k8 = K / 8u;
  let base_a = r * k8;
  let base_b = (r + N) * k8;

  var acc_a = 0.0;
  var acc_b = 0.0;
  var ss = 0.0;
  for (var i = lane; i < k8; i += 32u) {
    var xa = x[2u * i];
    var xb = x[2u * i + 1u];
    if (NORM) {
      ss += dot(xa, xa) + dot(xb, xb);
      xa *= gamma[2u * i];
      xb *= gamma[2u * i + 1u];
    }
    acc_a += dot8(W[base_a + i], xa, xb);
    if (MODE == 2u) {
      acc_b += dot8(W[base_b + i], xa, xb);
    }
  }

  red_a[tid] = acc_a;
  red_b[tid] = acc_b;
  red_s[tid] = ss;
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if (lane < s) {
      red_a[tid] += red_a[tid + s];
      red_b[tid] += red_b[tid + s];
      red_s[tid] += red_s[tid + s];
    }
    workgroupBarrier();
  }

  if (lane == 0u && valid) {
    var a = red_a[tid];
    var b = red_b[tid];
    if (NORM) {
      let inv = inverseSqrt(red_s[tid] / f32(K) + EPS);
      a *= inv;
      b *= inv;
    }
    if (MODE == 2u) {
      out[row] = a / (1.0 + exp(-a)) * b;
    } else {
      if (HAS_BIAS) {
        a += bias[row];
      }
      if (MODE == 1u) {
        out[row] += a;
      } else {
        out[row] = a;
      }
    }
  }
}
