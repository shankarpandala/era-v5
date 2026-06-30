// Minimal 2D PCA via power iteration on the covariance matrix.
//
// Used to project learned token embeddings (dim 8–16) down to a plane so the
// emergent category clusters in S1-3 can be seen. Rows = samples, cols = dims.

function covariance(rows, mean) {
  const n = rows.length
  const dim = mean.length
  const C = Array.from({ length: dim }, () => new Float64Array(dim))
  for (const r of rows) {
    for (let i = 0; i < dim; i++) {
      const di = r[i] - mean[i]
      for (let j = i; j < dim; j++) {
        C[i][j] += (di * (r[j] - mean[j])) / n
      }
    }
  }
  for (let i = 0; i < dim; i++) for (let j = i + 1; j < dim; j++) C[j][i] = C[i][j]
  return C
}

function matVec(C, v) {
  const out = new Float64Array(v.length)
  for (let i = 0; i < C.length; i++) {
    let s = 0
    for (let j = 0; j < v.length; j++) s += C[i][j] * v[j]
    out[i] = s
  }
  return out
}

function normalize(v) {
  let n = 0
  for (let i = 0; i < v.length; i++) n += v[i] * v[i]
  n = Math.sqrt(n) || 1
  for (let i = 0; i < v.length; i++) v[i] /= n
  return v
}

// Dominant eigenvector of C via power iteration.
function topEigenvector(C, rng, iters = 200) {
  const dim = C.length
  let v = new Float64Array(dim)
  for (let i = 0; i < dim; i++) v[i] = rng.gaussian()
  v = normalize(v)
  for (let k = 0; k < iters; k++) v = normalize(matVec(C, v))
  return v
}

// Project rows (array of number[]) onto their top 2 principal components.
// Returns array of [pc1, pc2]. rng makes the (sign-ambiguous) axes reproducible.
export function pca2d(rows, rng) {
  const dim = rows[0].length
  const mean = new Float64Array(dim)
  for (const r of rows) for (let i = 0; i < dim; i++) mean[i] += r[i] / rows.length

  const C = covariance(rows, mean)
  const v1 = topEigenvector(C, rng)

  // Deflate C by removing the first component, then take the next eigenvector.
  const lambda1 = dot(matVec(C, v1), v1)
  const C2 = C.map((row, i) => Float64Array.from(row, (val, j) => val - lambda1 * v1[i] * v1[j]))
  const v2 = topEigenvector(C2, rng)

  return rows.map((r) => {
    let a = 0
    let b = 0
    for (let i = 0; i < dim; i++) {
      const c = r[i] - mean[i]
      a += c * v1[i]
      b += c * v2[i]
    }
    return [a, b]
  })
}

function dot(a, b) {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}
