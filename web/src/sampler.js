// Greedy / temperature / top-p sampling (same algorithm as common/sampler.h).
export class Sampler {
  constructor({ temperature = 0.6, topP = 0.9, seed = 1234 } = {}) {
    this.temperature = temperature;
    this.topP = topP;
    this.state = BigInt(seed || 1);
  }

  get greedy() {
    return this.temperature <= 0;
  }

  uniform() {
    // xorshift64*, top 24 bits
    let s = this.state;
    s ^= s >> 12n;
    s ^= (s << 25n) & 0xffffffffffffffffn;
    s ^= s >> 27n;
    this.state = s;
    const r = (s * 0x2545f4914f6cdd1dn) & 0xffffffffffffffffn;
    return Number(r >> 40n) / 16777216;
  }

  sample(logits) {
    const n = logits.length;
    if (this.greedy) return argmax(logits);
    if (n <= 1) return 0;
    // NaN / +inf logits mean something upstream overflowed: report it instead of sampling garbage.
    let mx = -Infinity, best = 0;
    for (let i = 0; i < n; i++) {
      const v = logits[i];
      if (Number.isNaN(v)) throw new Error("sampler: the logits contain NaN");
      if (v > mx) { mx = v; best = i; }
    }
    if (mx === Infinity) throw new Error("sampler: the logits contain +inf");
    if (mx === -Infinity) throw new Error("sampler: every logit is -inf");
    const probs = new Float64Array(n);
    let sum = 0;
    for (let i = 0; i < n; i++) {
      probs[i] = Math.exp((logits[i] - mx) / this.temperature);
      sum += probs[i];
    }
    for (let i = 0; i < n; i++) probs[i] /= sum;
    const r = this.uniform();
    if (this.topP <= 0 || this.topP >= 1) {
      let cdf = 0;
      for (let i = 0; i < n; i++) {
        cdf += probs[i];
        if (r < cdf) return i;
      }
      return n - 1;
    }
    // Same nucleus pre-filter as the C++ sampler; the top token is always kept so `cand` is never empty.
    const cutoff = (1 - this.topP) / (n - 1);
    const cand = [];
    for (let i = 0; i < n; i++) if (probs[i] >= cutoff || i === best) cand.push(i);
    cand.sort((a, b) => probs[b] - probs[a] || a - b);
    let cum = 0, last = cand.length - 1;
    for (let i = 0; i < cand.length; i++) {
      cum += probs[cand[i]];
      if (cum > this.topP) { last = i; break; }
    }
    const target = r * cum;
    let cdf = 0;
    for (let i = 0; i <= last; i++) {
      cdf += probs[cand[i]];
      if (target < cdf) return cand[i];
    }
    return cand[last];
  }
}

export function argmax(x) {
  let best = 0;
  for (let i = 1; i < x.length; i++) if (x[i] > x[best]) best = i;
  return best;
}
