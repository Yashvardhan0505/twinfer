// Decode attention, one workgroup per query head (kv head = h / (H / HKV)).
// scores -> softmax (workgroup reductions) -> o = sum p_j v_j. For PV the invocations split into
// slices of HD/2, each owning one packed pair of dims, then the slices get summed.
// TODO: split-KV like the CUDA version, this only launches H workgroups.

struct Params {
  token: u32,
  pos: u32,
  pad0: u32,
  pad1: u32,
}

override H: u32;
override HKV: u32;
override HD: u32;
override MAXT: u32;

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> kc: array<u32>;
@group(0) @binding(2) var<storage, read> vc: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> scores: array<f32>;  // [H, MAXT] scratch
@group(0) @binding(5) var<storage, read_write> out: array<f32>;     // [H * HD]

var<workgroup> red: array<f32, 256>;
var<workgroup> part: array<vec2<f32>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let h = wg.x;
  let g = h / (H / HKV);
  let len = params.pos + 1u;
  let half = HD / 2u;
  let scale = inverseSqrt(f32(HD));
  let qb = h * HD;
  let sb = h * MAXT;

  var mx = -3.0e38;
  for (var j = tid; j < len; j += 256u) {
    let kb = (g * MAXT + j) * half;
    var s = 0.0;
    for (var i = 0u; i < half; i++) {
      let kv = unpack2x16float(kc[kb + i]);
      s += q[qb + 2u * i] * kv.x + q[qb + 2u * i + 1u] * kv.y;
    }
    s *= scale;
    scores[sb + j] = s;
    mx = max(mx, s);
  }
  red[tid] = mx;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (tid < s) {
      red[tid] = max(red[tid], red[tid + s]);
    }
    workgroupBarrier();
  }
  let m = red[0];
  workgroupBarrier();

  var sum = 0.0;
  for (var j = tid; j < len; j += 256u) {
    let p = exp(scores[sb + j] - m);
    scores[sb + j] = p;
    sum += p;
  }
  red[tid] = sum;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (tid < s) {
      red[tid] += red[tid + s];
    }
    workgroupBarrier();
  }
  let l = red[0];
  storageBarrier();  // probabilities written by other invocations are read below

  let slices = 256u / half;
  let slice = tid / half;
  let pi = tid % half;
  var o = vec2<f32>(0.0, 0.0);
  for (var j = slice; j < len; j += slices) {
    o += scores[sb + j] * unpack2x16float(vc[(g * MAXT + j) * half + pi]);
  }
  part[tid] = o;
  workgroupBarrier();
  if (tid < half) {
    var t = vec2<f32>(0.0, 0.0);
    for (var sl = 0u; sl < slices; sl++) {
      t += part[sl * half + tid];
    }
    out[qb + 2u * tid] = t.x / l;
    out[qb + 2u * tid + 1u] = t.y / l;
  }
}
