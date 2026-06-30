import { useCallback, useEffect, useRef, useState } from 'react'

// Generic rAF-chunked training driver.
//
// A claim supplies a `build()` that returns a fresh "trainer" object:
//   { step(): void, snapshot(): object, model?: ... }
// step() runs ONE epoch; snapshot() returns whatever metrics the UI shows.
//
// Pacing is EASE-IN: the early epochs — where the decision boundary changes the
// most — run slowly (< 1 epoch/frame, so the morph is clearly visible), then the
// rate ramps up exponentially to blow through the boring fine-tuning tail. A
// fractional accumulator lets the rate dip below 1 epoch/frame. A hard time cap
// per frame keeps the heavy S1-4 (n=2000) case from freezing the tab. The loop
// rebuilds and (optionally) restarts whenever `deps` change — that's how a knob
// re-runs the experiment.
const MAX_FRAME_MS = 34 // safety cap so a frame can't lock the UI
const RATE_MIN = 0.5 // epochs/frame at the start — slow, dramatic
const RATE_MAX = 20 // epochs/frame cap for the tail — fast
const RATE_HALFLIFE = 45 // epochs to double the rate

// Epochs to advance this frame, as a function of how far training has progressed.
function rateAt(epoch) {
  return Math.min(RATE_MAX, RATE_MIN * Math.pow(2, epoch / RATE_HALFLIFE))
}

export default function useTrainer({ build, epochs, autoStart = true, deps = [] }) {
  const trainerRef = useRef(null)
  const rafRef = useRef(0)
  const epochRef = useRef(0)
  const runningRef = useRef(false)
  const accRef = useRef(0) // fractional epoch accumulator for sub-1-epoch/frame pacing

  const [epoch, setEpoch] = useState(0)
  const [snapshot, setSnapshot] = useState(null)
  const [running, setRunning] = useState(false)
  const [tick, setTick] = useState(0) // bumps once per frame batch → redraw signal

  const stop = useCallback(() => {
    runningRef.current = false
    setRunning(false)
    cancelAnimationFrame(rafRef.current)
  }, [])

  const publish = useCallback(() => {
    setEpoch(epochRef.current)
    setSnapshot(trainerRef.current?.snapshot() ?? null)
    setTick((t) => t + 1)
  }, [])

  const loop = useCallback(() => {
    const t = trainerRef.current
    if (!t || !runningRef.current) return
    const t0 = performance.now()
    // Add this frame's quota (may be < 1 early on); step whole epochs out of it.
    accRef.current += rateAt(epochRef.current)
    while (accRef.current >= 1 && epochRef.current < epochs && performance.now() - t0 < MAX_FRAME_MS) {
      t.step()
      epochRef.current += 1
      accRef.current -= 1
    }
    publish()
    if (epochRef.current < epochs) {
      rafRef.current = requestAnimationFrame(loop)
    } else {
      runningRef.current = false
      setRunning(false)
    }
  }, [epochs, publish])

  const start = useCallback(() => {
    if (runningRef.current || epochRef.current >= epochs) return
    accRef.current = 0
    runningRef.current = true
    setRunning(true)
    rafRef.current = requestAnimationFrame(loop)
  }, [epochs, loop])

  const reset = useCallback(
    (start = false) => {
      cancelAnimationFrame(rafRef.current)
      trainerRef.current = build()
      epochRef.current = 0
      accRef.current = 0
      runningRef.current = false
      setEpoch(0)
      setSnapshot(trainerRef.current?.snapshot() ?? null)
      setTick((t) => t + 1)
      if (start) {
        runningRef.current = true
        setRunning(true)
        rafRef.current = requestAnimationFrame(loop)
      } else {
        setRunning(false)
      }
    },
    [build, loop]
  )

  // Rebuild + restart whenever the experiment configuration changes.
  useEffect(() => {
    reset(autoStart)
    return () => cancelAnimationFrame(rafRef.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return { epoch, snapshot, running, tick, trainer: trainerRef, start, stop, reset, epochs }
}
