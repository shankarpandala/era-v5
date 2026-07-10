// From-scratch byte-level BPE ENCODER — a byte-for-byte mirror of the Python
// reference (tokenizer/bpe.py). It reads the shipped tokenizer.json (pattern +
// ordered merges) and reproduces the exact token stream the grader's Python
// produces, so every number the widget shows is recomputed live from the same
// artifact a grader downloads. Proven equal by scripts/check_parity.mjs.
//
// Pure module: no browser or Node specifics beyond TextEncoder (available in
// both). Encoding operates on UTF-8 byte integers, so it is script-agnostic.

// GPT-2 reversible byte<->unicode map — used ONLY to render raw bytes as
// visible characters (vocab list, token chips); it plays no role in encoding.
function bytesToUnicode() {
  const bs = []
  for (let i = 33; i <= 126; i++) bs.push(i) // ! .. ~
  for (let i = 161; i <= 172; i++) bs.push(i) // ¡ .. ¬
  for (let i = 174; i <= 255; i++) bs.push(i) // ® .. ÿ
  const cs = bs.slice()
  let n = 0
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b)
      cs.push(256 + n)
      n++
    }
  }
  const map = new Array(256)
  for (let i = 0; i < bs.length; i++) map[bs[i]] = String.fromCodePoint(cs[i])
  return map
}

export const BYTE_ENCODER = bytesToUnicode()

/** Render a token's raw bytes (array of ints) as a printable string. */
export function renderToken(bytes) {
  let s = ''
  for (const b of bytes) s += BYTE_ENCODER[b]
  return s
}

const _enc = new TextEncoder()

export class BPE {
  constructor(pattern, merges) {
    this.pattern = pattern
    this.merges = merges // Array<[a, b]>, index == rank
    this.ranks = new Map()
    for (let i = 0; i < merges.length; i++) {
      this.ranks.set(merges[i][0] * 65536 + merges[i][1], i)
    }
    // Reconstruct id -> raw bytes for rendering / decoding.
    this.vocab = new Array(256 + merges.length)
    for (let i = 0; i < 256; i++) this.vocab[i] = [i]
    for (let i = 0; i < merges.length; i++) {
      const [a, b] = merges[i]
      this.vocab[256 + i] = this.vocab[a].concat(this.vocab[b])
    }
  }

  static fromJSON(data) {
    return new BPE(data.pattern, data.merges)
  }

  get vocabSize() {
    return 256 + this.merges.length
  }

  _mergeSeq(ids, a, b, newId) {
    const out = []
    let i = 0
    const n = ids.length
    while (i < n) {
      if (i < n - 1 && ids[i] === a && ids[i + 1] === b) {
        out.push(newId)
        i += 2
      } else {
        out.push(ids[i])
        i += 1
      }
    }
    return out
  }

  _encodeChunk(ids) {
    while (ids.length >= 2) {
      let bestRank = Infinity
      let bestA = -1
      let bestB = -1
      for (let i = 0; i < ids.length - 1; i++) {
        const r = this.ranks.get(ids[i] * 65536 + ids[i + 1])
        if (r !== undefined && r < bestRank) {
          bestRank = r
          bestA = ids[i]
          bestB = ids[i + 1]
        }
      }
      if (bestRank === Infinity) break
      ids = this._mergeSeq(ids, bestA, bestB, 256 + bestRank)
    }
    return ids
  }

  /** Fresh regex each call — matchAll needs the global flag and a clean lastIndex. */
  _re() {
    return new RegExp(this.pattern, 'gu')
  }

  /** Encode text -> array of token ids. */
  encode(text) {
    const out = []
    for (const m of text.matchAll(this._re())) {
      const bytes = Array.from(_enc.encode(m[0]))
      const ids = this._encodeChunk(bytes)
      for (const id of ids) out.push(id)
    }
    return out
  }

  /** Encode -> array of { id, text } for display (token chips). */
  encodeToTokens(text) {
    const toks = []
    for (const m of text.matchAll(this._re())) {
      const bytes = Array.from(_enc.encode(m[0]))
      for (const id of this._encodeChunk(bytes)) {
        toks.push({ id, text: renderToken(this.vocab[id]) })
      }
    }
    return toks
  }
}

/** Word count: maximal non-whitespace runs — matches Python len(text.split()). */
export function wordCount(text) {
  const t = text.trim()
  if (!t) return 0
  return t.split(/\s+/).length
}
