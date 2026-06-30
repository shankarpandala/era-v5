// Seeded pseudo-random number generator (mulberry32).
//
// Every source of randomness in the app — dataset generation, weight
// initialization, mini-batch shuffling — routes through one of these so a
// given seed reproduces exactly the same run. Never call Math.random() in
// lib/nn.js or lib/datasets.js; thread an RNG instance through instead.

export function makeRng(seed = 1) {
  let a = seed >>> 0
  // Spread tiny seeds (1, 2, 3…) across the state space so nearby seeds give
  // visibly different runs.
  a = (a + 0x6d2b79f5) >>> 0

  const rng = {
    // Uniform float in [0, 1).
    next() {
      a |= 0
      a = (a + 0x6d2b79f5) | 0
      let t = Math.imul(a ^ (a >>> 15), 1 | a)
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296
    },
    // Uniform float in [min, max).
    uniform(min, max) {
      return min + (max - min) * rng.next()
    },
    // Integer in [0, n).
    int(n) {
      return Math.floor(rng.next() * n)
    },
    // Standard normal via Box–Muller.
    gaussian(mean = 0, std = 1) {
      let u = 0
      let v = 0
      while (u === 0) u = rng.next()
      while (v === 0) v = rng.next()
      const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
      return mean + std * z
    },
    // In-place Fisher–Yates shuffle.
    shuffle(arr) {
      for (let i = arr.length - 1; i > 0; i--) {
        const j = rng.int(i + 1)
        const tmp = arr[i]
        arr[i] = arr[j]
        arr[j] = tmp
      }
      return arr
    },
  }
  return rng
}
