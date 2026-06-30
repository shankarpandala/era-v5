// A tiny, dependency-free MLP with hand-written backprop.
//
// Everything the four claims need runs on this: Dense layers, an optional ReLU
// between them, a sigmoid+BCE head for binary tasks and a softmax+cross-entropy
// head for multi-class / next-token tasks, plus Adam and plain SGD. Models are
// small (≤2000 points, ≤5 layers), so a readable matrix implementation on
// Float64Array is plenty fast — including pushing a 200×200 decision-boundary
// grid through forward() in one batched call.
//
// Matrices are plain objects { r, c, d } where d is a row-major Float64Array.

export function mat(r, c, fill = 0) {
  const d = new Float64Array(r * c)
  if (fill !== 0) d.fill(fill)
  return { r, c, d }
}

// C = A (r×k) · B (k×n)
export function matmul(A, B) {
  if (A.c !== B.r) throw new Error(`matmul shape ${A.r}x${A.c} · ${B.r}x${B.c}`)
  const r = A.r
  const k = A.c
  const n = B.c
  const out = mat(r, n)
  const a = A.d
  const b = B.d
  const o = out.d
  for (let i = 0; i < r; i++) {
    const ai = i * k
    const oi = i * n
    for (let p = 0; p < k; p++) {
      const av = a[ai + p]
      if (av === 0) continue
      const bp = p * n
      for (let j = 0; j < n; j++) o[oi + j] += av * b[bp + j]
    }
  }
  return out
}

// Chain-multiply a list of matrices left to right: W1·W2·…·Wn.
export function matChain(mats) {
  return mats.reduce((acc, m) => (acc ? matmul(acc, m) : m))
}

function heInit(inDim, outDim, rng) {
  // He initialization keeps ReLU activations from collapsing or exploding.
  const std = Math.sqrt(2 / inDim)
  const W = mat(inDim, outDim)
  for (let i = 0; i < W.d.length; i++) W.d[i] = rng.gaussian(0, std)
  return W
}

class Dense {
  constructor(inDim, outDim, rng, { bias = true } = {}) {
    this.inDim = inDim
    this.outDim = outDim
    this.W = heInit(inDim, outDim, rng)
    this.b = bias ? mat(1, outDim) : null
    this._initAdam()
  }

  _initAdam() {
    this.mW = new Float64Array(this.W.d.length)
    this.vW = new Float64Array(this.W.d.length)
    if (this.b) {
      this.mb = new Float64Array(this.b.d.length)
      this.vb = new Float64Array(this.b.d.length)
    }
  }

  forward(X) {
    this.X = X
    const Z = matmul(X, this.W)
    if (this.b) {
      for (let i = 0; i < Z.r; i++) {
        const zi = i * Z.c
        for (let j = 0; j < Z.c; j++) Z.d[zi + j] += this.b.d[j]
      }
    }
    return Z
  }

  // dY: gradient w.r.t. this layer's output. Returns gradient w.r.t. input.
  backward(dY) {
    const X = this.X
    // dW = Xᵀ · dY
    this.dW = mat(this.inDim, this.outDim)
    for (let i = 0; i < X.r; i++) {
      const xi = i * X.c
      const yi = i * dY.c
      for (let p = 0; p < this.inDim; p++) {
        const xv = X.d[xi + p]
        if (xv === 0) continue
        const wp = p * this.outDim
        for (let j = 0; j < this.outDim; j++) this.dW.d[wp + j] += xv * dY.d[yi + j]
      }
    }
    if (this.b) {
      this.db = new Float64Array(this.outDim)
      for (let i = 0; i < dY.r; i++) {
        const yi = i * dY.c
        for (let j = 0; j < this.outDim; j++) this.db[j] += dY.d[yi + j]
      }
    }
    // dX = dY · Wᵀ
    const dX = mat(X.r, this.inDim)
    for (let i = 0; i < dY.r; i++) {
      const yi = i * dY.c
      const xi = i * this.inDim
      for (let j = 0; j < this.outDim; j++) {
        const g = dY.d[yi + j]
        if (g === 0) continue
        for (let p = 0; p < this.inDim; p++) dX.d[xi + p] += g * this.W.d[p * this.outDim + j]
      }
    }
    return dX
  }

  step(opt, t) {
    opt.update(this.W.d, this.dW.d, this.mW, this.vW, t)
    if (this.b) opt.update(this.b.d, this.db, this.mb, this.vb, t)
  }

  get params() {
    return this.b ? this.W.d.length + this.b.d.length : this.W.d.length
  }
}

class ReLU {
  forward(X) {
    this.mask = new Uint8Array(X.d.length)
    const out = mat(X.r, X.c)
    for (let i = 0; i < X.d.length; i++) {
      if (X.d[i] > 0) {
        out.d[i] = X.d[i]
        this.mask[i] = 1
      }
    }
    return out
  }

  backward(dY) {
    const dX = mat(dY.r, dY.c)
    for (let i = 0; i < dY.d.length; i++) dX.d[i] = this.mask[i] ? dY.d[i] : 0
    return dX
  }

  step() {}
  get params() {
    return 0
  }
}

// --- optimizers ----------------------------------------------------------

export class Adam {
  constructor({ lr = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8 } = {}) {
    this.lr = lr
    this.b1 = beta1
    this.b2 = beta2
    this.eps = eps
  }
  update(p, g, m, v, t) {
    const b1 = this.b1
    const b2 = this.b2
    const bc1 = 1 - Math.pow(b1, t)
    const bc2 = 1 - Math.pow(b2, t)
    for (let i = 0; i < p.length; i++) {
      m[i] = b1 * m[i] + (1 - b1) * g[i]
      v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i]
      const mh = m[i] / bc1
      const vh = v[i] / bc2
      p[i] -= (this.lr * mh) / (Math.sqrt(vh) + this.eps)
    }
  }
}

export class SGD {
  constructor({ lr = 0.1 } = {}) {
    this.lr = lr
  }
  update(p, g) {
    for (let i = 0; i < p.length; i++) p[i] -= this.lr * g[i]
  }
}

// --- model ---------------------------------------------------------------

// spec: { inDim, hidden: [n1, n2, ...], activation: 'relu'|'none',
//         outDim, head: 'sigmoid'|'softmax', bias, rng }
// hidden:[] + head:'sigmoid' + activation:'none' => plain logistic regression.
export class MLP {
  constructor(spec) {
    const { inDim, hidden = [], activation = 'relu', outDim, head, rng, bias = true } = spec
    this.head = head
    this.layers = []
    let dim = inDim
    for (const h of hidden) {
      this.layers.push(new Dense(dim, h, rng, { bias }))
      if (activation === 'relu') this.layers.push(new ReLU())
      dim = h
    }
    this.layers.push(new Dense(dim, outDim, rng, { bias }))
    this.t = 0
  }

  // Returns raw logits matrix (rows × outDim). Used directly for boundary grids.
  forward(X) {
    let h = X
    for (const layer of this.layers) h = layer.forward(h)
    return h
  }

  // Probabilities: sigmoid (outDim 1) or row-wise softmax.
  predictProba(X) {
    const Z = this.forward(X)
    return this.head === 'softmax' ? softmaxRows(Z) : sigmoid(Z)
  }

  // One full-batch gradient step. Y is an array: 0/1 for sigmoid, class index
  // for softmax. Returns mean loss.
  trainStep(X, Y, opt) {
    this.t += 1
    const Z = this.forward(X)
    let loss = 0
    const dZ = mat(Z.r, Z.c)
    const invN = 1 / Z.r
    if (this.head === 'sigmoid') {
      for (let i = 0; i < Z.r; i++) {
        const p = sig(Z.d[i])
        const y = Y[i]
        loss += -(y * Math.log(p + 1e-12) + (1 - y) * Math.log(1 - p + 1e-12))
        dZ.d[i] = (p - y) * invN
      }
    } else {
      for (let i = 0; i < Z.r; i++) {
        const off = i * Z.c
        let max = -Infinity
        for (let j = 0; j < Z.c; j++) if (Z.d[off + j] > max) max = Z.d[off + j]
        let sum = 0
        for (let j = 0; j < Z.c; j++) sum += Math.exp(Z.d[off + j] - max)
        const y = Y[i]
        for (let j = 0; j < Z.c; j++) {
          const p = Math.exp(Z.d[off + j] - max) / sum
          dZ.d[off + j] = (p - (j === y ? 1 : 0)) * invN
        }
        loss += -(Z.d[off + y] - max - Math.log(sum))
      }
    }
    let g = dZ
    for (let i = this.layers.length - 1; i >= 0; i--) g = this.layers[i].backward(g)
    for (const layer of this.layers) layer.step(opt, this.t)
    return loss * invN
  }

  // 0/1 accuracy against labels (class index, or 0/1 for sigmoid).
  accuracy(X, Y) {
    const P = this.predictProba(X)
    let correct = 0
    if (this.head === 'sigmoid') {
      for (let i = 0; i < P.r; i++) if ((P.d[i] >= 0.5 ? 1 : 0) === Y[i]) correct++
    } else {
      for (let i = 0; i < P.r; i++) {
        const off = i * P.c
        let arg = 0
        let best = -Infinity
        for (let j = 0; j < P.c; j++)
          if (P.d[off + j] > best) {
            best = P.d[off + j]
            arg = j
          }
        if (arg === Y[i]) correct++
      }
    }
    return correct / P.r
  }

  // Mean loss without a gradient step (for held-out evaluation).
  loss(X, Y) {
    const Z = this.forward(X)
    let loss = 0
    if (this.head === 'sigmoid') {
      for (let i = 0; i < Z.r; i++) {
        const p = sig(Z.d[i])
        const y = Y[i]
        loss += -(y * Math.log(p + 1e-12) + (1 - y) * Math.log(1 - p + 1e-12))
      }
    } else {
      for (let i = 0; i < Z.r; i++) {
        const off = i * Z.c
        let max = -Infinity
        for (let j = 0; j < Z.c; j++) if (Z.d[off + j] > max) max = Z.d[off + j]
        let sum = 0
        for (let j = 0; j < Z.c; j++) sum += Math.exp(Z.d[off + j] - max)
        loss += -(Z.d[off + Y[i]] - max - Math.log(sum))
      }
    }
    return loss / Z.r
  }

  // Weight matrices of the Dense layers, in order (for the S1-2 collapse demo).
  denseWeights() {
    return this.layers.filter((l) => l instanceof Dense).map((l) => l.W)
  }

  totalParams() {
    return this.layers.reduce((s, l) => s + l.params, 0)
  }
}

// --- activations ---------------------------------------------------------

function sig(x) {
  return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x))
}

export function sigmoid(Z) {
  const out = mat(Z.r, Z.c)
  for (let i = 0; i < Z.d.length; i++) out.d[i] = sig(Z.d[i])
  return out
}

export function softmaxRows(Z) {
  const out = mat(Z.r, Z.c)
  for (let i = 0; i < Z.r; i++) {
    const off = i * Z.c
    let max = -Infinity
    for (let j = 0; j < Z.c; j++) if (Z.d[off + j] > max) max = Z.d[off + j]
    let sum = 0
    for (let j = 0; j < Z.c; j++) {
      const e = Math.exp(Z.d[off + j] - max)
      out.d[off + j] = e
      sum += e
    }
    for (let j = 0; j < Z.c; j++) out.d[off + j] /= sum
  }
  return out
}
