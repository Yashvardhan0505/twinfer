// RoPE on q and k (HF rotate_half: i pairs with i + HD/2), then append k/v to the packed fp16
// cache [HKV, MAXT, HD/2]. One workgroup per head; invocation j writes elements 2j and 2j+1
// so each cache write is a single u32.

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

@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> cos_t: array<f32>;
@group(0) @binding(2) var<storage, read> sin_t: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> q: array<f32>;
@group(0) @binding(5) var<storage, read_write> kc: array<u32>;
@group(0) @binding(6) var<storage, read_write> vc: array<u32>;

fn rotated(base: u32, e: u32, pos: u32) -> f32 {
  let half = HD / 2u;
  if (e < half) {
    return qkv[base + e] * cos_t[pos * half + e] - qkv[base + e + half] * sin_t[pos * half + e];
  }
  let i = e - half;
  return qkv[base + e] * cos_t[pos * half + i] + qkv[base + i] * sin_t[pos * half + i];
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) j: u32) {
  let half = HD / 2u;
  if (j >= half) {
    return;
  }
  let head = wg.x;
  let pos = params.pos;
  let base = head * HD;
  let e0 = 2u * j;
  let e1 = 2u * j + 1u;
  if (head < H) {
    q[base + e0] = rotated(base, e0, pos);
    q[base + e1] = rotated(base, e1, pos);
  } else if (head < H + HKV) {
    let g = head - H;
    kc[(g * MAXT + pos) * half + j] = pack2x16float(vec2<f32>(rotated(base, e0, pos), rotated(base, e1, pos)));
  } else {
    let g = head - H - HKV;
    vc[(g * MAXT + pos) * half + j] = pack2x16float(vec2<f32>(qkv[base + e0], qkv[base + e1]));
  }
}
