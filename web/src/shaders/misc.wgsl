// Small kernels: embedding lookup, argmax, and in-place weight conversion to packed fp16.

struct Params {
  token: u32,
  pos: u32,
  pad0: u32,
  pad1: u32,
}

override D: u32 = 1u;
override COUNT: u32 = 1u;  // number of u32 words to convert / logits to scan

// embedding: x = E[token] (E packed fp16 [V, D/2])
@group(0) @binding(0) var<storage, read> E: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(256)
fn embed(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= D / 2u) {
    return;
  }
  let v = unpack2x16float(E[params.token * (D / 2u) + i]);
  x[2u * i] = v.x;
  x[2u * i + 1u] = v.y;
}

// argmax over COUNT logits (ties -> lowest index)
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<u32>;
var<workgroup> best_v: array<f32, 256>;
var<workgroup> best_i: array<u32, 256>;

@compute @workgroup_size(256)
fn argmax(@builtin(local_invocation_index) tid: u32) {
  var bv = -3.4e38;
  var bi = 0xffffffffu;
  for (var i = tid; i < COUNT; i += 256u) {
    let v = logits[i];
    if (v > bv) {
      bv = v;
      bi = i;
    }
  }
  best_v[tid] = bv;
  best_i[tid] = bi;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (tid < s) {
      let ov = best_v[tid + s];
      let oi = best_i[tid + s];
      if (ov > best_v[tid] || (ov == best_v[tid] && oi < best_i[tid])) {
        best_v[tid] = ov;
        best_i[tid] = oi;
      }
    }
    workgroupBarrier();
  }
  if (tid == 0u) {
    result[0] = best_i[0];
  }
}

// weight conversion
// bf16 pairs -> fp16 pairs, in place (a bf16 is the top half of an fp32).
@group(0) @binding(0) var<storage, read_write> words: array<u32>;

@compute @workgroup_size(256)
fn bf16_to_f16(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.y * nwg.x * 256u + gid.x;
  if (i >= COUNT) {
    return;
  }
  let w = words[i];
  let lo = bitcast<f32>(w << 16u);
  let hi = bitcast<f32>(w & 0xffff0000u);
  words[i] = pack2x16float(clamp(vec2<f32>(lo, hi), vec2<f32>(-65504.0), vec2<f32>(65504.0)));
}

// fp32 -> packed fp16 (src has 2 * COUNT floats).
@group(0) @binding(0) var<storage, read> src32: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst16: array<u32>;

@compute @workgroup_size(256)
fn f32_to_f16(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.y * nwg.x * 256u + gid.x;
  if (i >= COUNT) {
    return;
  }
  dst16[i] = pack2x16float(clamp(src32[i], vec2<f32>(-65504.0), vec2<f32>(65504.0)));
}
