import { useCallback, useEffect, useRef, useState } from 'react'

// Generic rAF-chunked training driver.
//
// A claim supplies a `build()` that returns a fresh "trainer" object:
//   { step(): void, snapshot(): object, model?: ... }
// step() runs ONE epoch; snapshot() returns whatever metrics the UI shows.
// This hook runs `stepsPerFrame` epochs per animation frame so the convergence
// is watchable (1 epoch/frame ≈ 5s for 300 epochs at 60fps — slow enough to see
// the boundary morph). `stepsPerFrame` is read from a live ref, so the speed
// knob changes pacing mid-run WITHOUT restarting. A hard time cap per frame
// keeps a heavy model (S1-4 n=2000) from freezing the tab. It rebuilds and
// (optionally) restarts whenever `deps` change — that's how a knob re-runs the
// experiment.
const MAX_FRAME_MS = 34 // safety cap so a frame can't lock the UI

export default function useTrainer({ build, epochs, autoStart = true, deps = [], stepsPerFrame = 1 }) {
  const trainerRef = useRef(null)
  const rafRef = useRef(0)
  const epochRef = useRef(0)
  const runningRef = useRef(false)
  const stepsRef = useRef(stepsPerFrame)
  stepsRef.current = Math.max(1, stepsPerFrame) // live: speed changes apply without a restart

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
    let n = 0
    while (n < stepsRef.current && epochRef.current < epochs && performance.now() - t0 < MAX_FRAME_MS) {
      t.step()
      epochRef.current += 1
      n += 1
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
    runningRef.current = true
    setRunning(true)
    rafRef.current = requestAnimationFrame(loop)
  }, [epochs, loop])

  const reset = useCallback(
    (start = false) => {
      cancelAnimationFrame(rafRef.current)
      trainerRef.current = build()
      epochRef.current = 0
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
