import { useCallback, useEffect, useRef, useState } from 'react'

// Generic rAF-chunked training driver.
//
// A claim supplies a `build()` that returns a fresh "trainer" object:
//   { step(): void, snapshot(): object, model?: ... }
// step() runs ONE epoch; snapshot() returns whatever metrics the UI shows.
// This hook calls step() repeatedly inside a ~12ms time budget per animation
// frame so training animates and the page stays responsive, then publishes the
// latest snapshot. It rebuilds and (optionally) restarts whenever `deps` change
// — that's how turning a knob re-runs the experiment.
export default function useTrainer({ build, epochs, autoStart = true, deps = [], frameBudgetMs = 12 }) {
  const trainerRef = useRef(null)
  const rafRef = useRef(0)
  const epochRef = useRef(0)
  const runningRef = useRef(false)

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
    while (performance.now() - t0 < frameBudgetMs && epochRef.current < epochs) {
      t.step()
      epochRef.current += 1
    }
    publish()
    if (epochRef.current < epochs) {
      rafRef.current = requestAnimationFrame(loop)
    } else {
      runningRef.current = false
      setRunning(false)
    }
  }, [epochs, frameBudgetMs, publish])

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
