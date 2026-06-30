// Canvas helpers for decision-boundary heatmaps.
//
// A boundary is drawn by evaluating the model on a grid of pixels. The whole
// grid is pushed through MLP.predictProba in ONE batched matrix call, then each
// probability is mapped to an RGBA value and blitted with putImageData. This is
// why we use raw <canvas> and not SVG: a 180×180 grid is 32k cells — that many
// DOM nodes would melt the page, but it's one ImageData here.

import { mat } from './nn.js'

// Class colors (kept in sync with --class-0 / --class-1 in index.css).
export const CLASS0 = [239, 68, 68] // red
export const CLASS1 = [59, 130, 246] // blue

// Build the (grid*grid × 2) matrix of input coordinates spanning [min,max]².
export function makeGrid(min, max, grid) {
  const X = mat(grid * grid, 2)
  const step = (max - min) / (grid - 1)
  let k = 0
  for (let j = 0; j < grid; j++) {
    const y = max - j * step // top row = max so canvas y matches plot y
    for (let i = 0; i < grid; i++) {
      X.d[k * 2] = min + i * step
      X.d[k * 2 + 1] = y
      k++
    }
  }
  return X
}

// Render P(class 1) over the grid into an ImageData scaled to the canvas.
// proba: result of model.predictProba(grid) — a (grid*grid × 1) matrix.
export function drawHeatmap(ctx, proba, grid, size, alpha = 0.55) {
  const img = ctx.createImageData(grid, grid)
  for (let k = 0; k < grid * grid; k++) {
    const p = proba.d[k] // probability of class 1
    // Blend red(class0) → blue(class1). Mid-tones near the boundary read white.
    const r = Math.round(CLASS0[0] * (1 - p) + CLASS1[0] * p)
    const g = Math.round(CLASS0[1] * (1 - p) + CLASS1[1] * p)
    const b = Math.round(CLASS0[2] * (1 - p) + CLASS1[2] * p)
    const o = k * 4
    img.data[o] = r
    img.data[o + 1] = g
    img.data[o + 2] = b
    img.data[o + 3] = Math.round(255 * alpha)
  }
  // Blit the small grid onto an offscreen canvas, then scale up (nearest → we
  // want crisp cells, but smoothing reads nicer for a probability field).
  const off = document.createElement('canvas')
  off.width = grid
  off.height = grid
  off.getContext('2d').putImageData(img, 0, 0)
  ctx.imageSmoothingEnabled = true
  ctx.drawImage(off, 0, 0, size, size)
}

// Map a data coordinate to a canvas pixel (y is flipped).
export function toPixel(v, min, max, size) {
  return ((v - min) / (max - min)) * size
}
export function toPixelY(v, min, max, size) {
  return size - ((v - min) / (max - min)) * size
}
