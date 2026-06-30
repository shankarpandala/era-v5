import { Suspense, lazy, useCallback, useMemo, useState } from 'react'
import { BlockMath } from 'react-katex'
import ClaimCard from '../components/ClaimCard.jsx'
import ScatterCanvas from '../components/ui/ScatterCanvas.jsx'
import Slider from '../components/ui/Slider.jsx'
import Toggle from '../components/ui/Toggle.jsx'
import Button from '../components/ui/Button.jsx'
import ErrorBoundary from '../components/ErrorBoundary.jsx'
import useTrainer from '../hooks/useTrainer.js'
import { makeRng } from '../lib/rng.js'
import { makeRings } from '../lib/datasets.js'
import { MLP, Adam } from '../lib/nn.js'

const RingLift3D = lazy(() => import('../three/RingLift3D.jsx'))
const ACCENT = '#3b82f6'

function AccBadge({ label, acc, good }) {
  const pct = acc == null ? '—' : `${(acc * 100).toFixed(1)}%`
  return (
    <div className="rounded-lg bg-zinc-100 px-3 py-1.5 text-center dark:bg-zinc-800">
      <div className="text-[10px] uppercase tracking-wide text-zinc-500">{label}</div>
      <div className={`font-mono text-sm font-semibold ${good ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400'}`}>
        {pct}
      </div>
    </div>
  )
}

export default function S1RingActivation() {
  const [n, setN] = useState(300)
  const [noise, setNoise] = useState(0.18)
  const [hidden, setHidden] = useState(16)
  const [lr, setLr] = useState(0.05)
  const [epochsTarget, setEpochsTarget] = useState(300)
  const [seed, setSeed] = useState(1)
  const [useReLU, setUseReLU] = useState(true)
  const [show3D, setShow3D] = useState(false)

  const data = useMemo(() => makeRings(n, noise, makeRng(seed)), [n, noise, seed])

  const build = useCallback(() => {
    const linModel = new MLP({ inDim: 2, hidden: [], outDim: 1, head: 'sigmoid', rng: makeRng(seed * 100 + 1) })
    const reluModel = new MLP({
      inDim: 2,
      hidden: [hidden],
      activation: useReLU ? 'relu' : 'none',
      outDim: 1,
      head: 'sigmoid',
      rng: makeRng(seed * 100 + 2),
    })
    const optLin = new Adam({ lr })
    const optRelu = new Adam({ lr })
    const { X, y } = data
    return {
      linModel,
      reluModel,
      X,
      step() {
        linModel.trainStep(X, y, optLin)
        reluModel.trainStep(X, y, optRelu)
      },
      snapshot() {
        return { linAcc: linModel.accuracy(X, y), reluAcc: reluModel.accuracy(X, y) }
      },
    }
  }, [data, hidden, lr, useReLU, seed])

  const { tick, running, snapshot, trainer, epoch, start, stop, reset } = useTrainer({
    build,
    epochs: epochsTarget,
    deps: [data, hidden, lr, useReLU, seed, epochsTarget],
  })

  const lifted = useMemo(() => {
    const t = trainer.current
    if (!t) return []
    const logits = t.reluModel.forward(t.X)
    return data.points.map((p, i) => ({
      x: p.x,
      y: p.y,
      label: p.label,
      z: Math.max(-4, Math.min(4, logits.d[i] * 0.5)),
    }))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick, data])

  return (
    <ClaimCard
      id="s1-1"
      code="S1-1"
      accent={ACCENT}
      title="Activations exist for a reason"
      claim="A model with no nonlinearity can only draw a straight boundary, so it cannot separate two concentric rings. Adding a single ReLU hidden layer can — only the activation changed."
      takeaway="The linear model is stuck near chance with a straight cut no matter how long it trains. Flip the activation on and the same-sized network wraps the ring to ~99%. The decision-boundary picture is the proof: nonlinearity is what buys curved boundaries."
    >
      <div className="panel p-4">
        <div className="grid gap-6 lg:grid-cols-[1fr_280px]">
          {/* Visualization */}
          <div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div>
                <ScatterCanvas
                  points={data.points}
                  domain={data.domain}
                  model={trainer.current?.linModel}
                  redrawKey={tick}
                  running={running}
                  title="Linear + sigmoid (no activation)"
                />
                <div className="mt-2">
                  <AccBadge label="train accuracy" acc={snapshot?.linAcc} good={false} />
                </div>
              </div>
              <div>
                <ScatterCanvas
                  points={data.points}
                  domain={data.domain}
                  model={trainer.current?.reluModel}
                  redrawKey={tick}
                  running={running}
                  title={useReLU ? '1 hidden layer + ReLU' : '1 hidden layer, NO activation'}
                />
                <div className="mt-2">
                  <AccBadge label="train accuracy" acc={snapshot?.reluAcc} good={useReLU} />
                </div>
              </div>
            </div>

            <div className="mt-4 flex flex-wrap items-center gap-3">
              <Button onClick={running ? stop : epoch >= epochsTarget ? () => reset(true) : start}>
                {running ? '⏸ Pause' : epoch >= epochsTarget ? '↻ Re-train' : '▶ Train'}
              </Button>
              <Button variant="ghost" onClick={() => reset(true)}>
                ↻ Restart
              </Button>
              <span className="font-mono text-xs text-zinc-500">
                epoch {epoch}/{epochsTarget}
              </span>
              <Button variant="ghost" onClick={() => setShow3D((s) => !s)} className="ml-auto">
                {show3D ? 'Hide' : 'Show'} 3D feature-lift
              </Button>
            </div>

            {show3D && (
              <div className="mt-4">
                <p className="mb-2 text-xs text-zinc-500 dark:text-zinc-400">
                  Each point is lifted to a height equal to the network's logit. The grey plane is the decision
                  surface (logit = 0). With ReLU on, the plane slips cleanly between the rings; turn it off and the
                  lift is a flat tilt — the rings stay interleaved and no plane can split them. Drag to orbit.
                </p>
                <ErrorBoundary>
                  <Suspense fallback={<div className="h-[340px] animate-pulse rounded-lg bg-zinc-800" />}>
                    <RingLift3D points={lifted} />
                  </Suspense>
                </ErrorBoundary>
              </div>
            )}
          </div>

          {/* Controls */}
          <div className="space-y-4">
            <Toggle
              label="ReLU activation"
              hint={useReLU ? 'nonlinearity ON' : 'collapses to linear'}
              checked={useReLU}
              onChange={setUseReLU}
            />
            <Slider label="noise" value={noise} min={0} max={0.5} step={0.01} onChange={setNoise} format={(v) => v.toFixed(2)} />
            <Slider label="points" value={n} min={60} max={600} step={20} onChange={setN} />
            <Slider label="hidden units" value={hidden} min={2} max={32} step={1} onChange={setHidden} />
            <Slider label="learning rate" value={lr} min={0.005} max={0.2} step={0.005} onChange={setLr} format={(v) => v.toFixed(3)} />
            <Slider label="epochs" value={epochsTarget} min={50} max={800} step={50} onChange={setEpochsTarget} />
            <Slider label="seed" value={seed} min={1} max={20} step={1} onChange={setSeed} />
            <div className="rounded-lg bg-zinc-50 p-2 text-[11px] text-zinc-500 dark:bg-zinc-800/60 dark:text-zinc-400">
              <BlockMath math={'\\hat y=\\sigma(Wx+b)\\;\\;\\text{vs}\\;\\;\\hat y=\\sigma\\big(W_2\\,\\mathrm{ReLU}(W_1x)\\big)'} />
            </div>
          </div>
        </div>
      </div>
    </ClaimCard>
  )
}
