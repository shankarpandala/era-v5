import { useCallback, useMemo, useState } from 'react'
import ClaimCard from '../components/ClaimCard.jsx'
import ScatterCanvas from '../components/ui/ScatterCanvas.jsx'
import LineChart from '../components/ui/LineChart.jsx'
import Slider from '../components/ui/Slider.jsx'
import Button from '../components/ui/Button.jsx'
import useTrainer from '../hooks/useTrainer.js'
import { makeGapExperiment } from '../lib/datasets.js'
import { makeRng } from '../lib/rng.js'
import { MLP, Adam } from '../lib/nn.js'

const ACCENT = '#f59e0b'
const SIZES = [20, 200, 2000]
const RECORD_EVERY = 4

export default function S1MemorizeGeneralize() {
  const [width, setWidth] = useState(24)
  const [lr, setLr] = useState(0.03)
  const [epochsTarget, setEpochsTarget] = useState(200)
  const [seed, setSeed] = useState(1)
  const [speed, setSpeed] = useState(1)
  const [selected, setSelected] = useState(20)

  const build = useCallback(() => {
    const bySize = {}
    for (const size of SIZES) {
      const exp = makeGapExperiment(size, seed)
      const model = new MLP({
        inDim: 2,
        hidden: [width, width],
        activation: 'relu',
        outDim: 1,
        head: 'sigmoid',
        rng: makeRng(seed * 100 + size),
      })
      bySize[size] = {
        model,
        opt: new Adam({ lr }),
        train: exp.train,
        test: exp.test,
        history: { train: [], test: [] },
      }
    }
    let e = 0
    return {
      bySize,
      step() {
        e += 1
        for (const size of SIZES) {
          const s = bySize[size]
          s.model.trainStep(s.train.X, s.train.y, s.opt)
        }
        if (e % RECORD_EVERY === 0) {
          for (const size of SIZES) {
            const s = bySize[size]
            s.history.train.push({ x: e, y: s.model.loss(s.train.X, s.train.y) })
            s.history.test.push({ x: e, y: s.model.loss(s.test.X, s.test.y) })
          }
        }
      },
      snapshot() {
        const histories = {}
        for (const size of SIZES) histories[size] = bySize[size].history
        return { histories }
      },
    }
  }, [width, lr, seed])

  const { tick, running, snapshot, trainer, epoch, start, stop, reset } = useTrainer({
    build,
    epochs: epochsTarget,
    stepsPerFrame: speed,
    deps: [width, lr, seed, epochsTarget],
  })

  const histories = snapshot?.histories
  const last = (size, which) => {
    const arr = histories?.[size]?.[which]
    return arr && arr.length ? arr[arr.length - 1].y : null
  }

  const gapSeries = useMemo(() => {
    const data = SIZES.map((size) => {
      const tr = last(size, 'train')
      const te = last(size, 'test')
      return { x: size, y: tr == null ? 0 : Math.max(0, te - tr) }
    })
    return [{ label: 'test − train loss', color: ACCENT, data, markers: true }]
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick])

  const curveSeries = useMemo(() => {
    const h = histories?.[selected]
    if (!h) return []
    return [
      { label: 'train loss', color: '#3b82f6', data: h.train },
      { label: 'test loss', color: '#ef4444', data: h.test, dashed: true },
    ]
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick, selected])

  const sel = trainer.current?.bySize?.[selected]

  return (
    <ClaimCard
      id="s1-4"
      code="S1-4"
      accent={ACCENT}
      title="Memorization vs generalization — and data closes the gap"
      claim="A high-capacity model on tiny data drives train loss to ~0 while held-out loss stays high. Growing the dataset closes the gap — the same network that memorized 20 points generalizes on 2000."
      takeaway="At n=20 the over-parameterized net fits every point (including label noise) and the boundary is a wiggly mess — train loss ≈ 0, test loss high. At n=2000 it can't memorize noise, the boundary smooths out, and the gap collapses. Capacity didn't change; data did. This is the course's 'data is everything.'"
    >
      <div className="panel p-4">
        <div className="grid gap-6 lg:grid-cols-[1fr_260px]">
          <div>
            <div className="mb-3 flex items-center gap-2">
              <span className="text-xs text-zinc-500">view boundary at n =</span>
              {SIZES.map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => setSelected(s)}
                  className={`rounded-md px-2.5 py-1 font-mono text-xs font-medium transition-colors ${
                    selected === s
                      ? 'bg-amber-500 text-white'
                      : 'bg-zinc-100 text-zinc-600 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300'
                  }`}
                >
                  {s}
                </button>
              ))}
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <div>
                <ScatterCanvas
                  points={sel?.train.points ?? []}
                  domain={sel?.train.domain ?? { min: -3.3, max: 3.3 }}
                  model={sel?.model}
                  redrawKey={`${tick}-${selected}`}
                  running={running}
                  size={260}
                  title={`Decision boundary (train set, n=${selected})`}
                />
                <div className="mt-2 grid grid-cols-2 gap-2 text-center text-xs">
                  <div className="rounded bg-zinc-100 py-1 dark:bg-zinc-800">
                    train loss{' '}
                    <span className="font-mono font-semibold text-blue-600 dark:text-blue-400">
                      {last(selected, 'train')?.toFixed(3) ?? '—'}
                    </span>
                  </div>
                  <div className="rounded bg-zinc-100 py-1 dark:bg-zinc-800">
                    test loss{' '}
                    <span className="font-mono font-semibold text-red-600 dark:text-red-400">
                      {last(selected, 'test')?.toFixed(3) ?? '—'}
                    </span>
                  </div>
                </div>
              </div>

              <div>
                <div className="mb-1 text-center text-xs font-medium text-zinc-600 dark:text-zinc-300">
                  Train vs test loss (n={selected})
                </div>
                <LineChart series={curveSeries} xLabel="epoch" yLabel="loss" yMin={0} height={232} />
              </div>
            </div>

            <div className="mt-4 rounded-lg bg-zinc-50 p-3 dark:bg-zinc-800/50">
              <div className="mb-1 text-center text-xs font-medium text-zinc-600 dark:text-zinc-300">
                Generalization gap vs dataset size — the money shot
              </div>
              <LineChart
                series={gapSeries}
                xLabel="training set size (log)"
                yLabel="gap"
                logX
                xMin={20}
                xMax={2000}
                yMin={0}
                height={200}
              />
            </div>

            <div className="mt-4 flex flex-wrap items-center gap-3">
              <Button onClick={running ? stop : epoch >= epochsTarget ? () => reset(true) : start}>
                {running ? '⏸ Pause' : epoch >= epochsTarget ? '↻ Re-train' : '▶ Train all sizes'}
              </Button>
              <Button variant="ghost" onClick={() => reset(true)}>
                ↻ Restart
              </Button>
              <span className="font-mono text-xs text-zinc-500">
                epoch {epoch}/{epochsTarget}
              </span>
            </div>
          </div>

          <div className="space-y-4">
            <Slider label="capacity (hidden width)" value={width} min={8} max={48} step={4} onChange={setWidth} />
            <Slider label="learning rate" value={lr} min={0.005} max={0.1} step={0.005} onChange={setLr} format={(v) => v.toFixed(3)} />
            <Slider label="epochs" value={epochsTarget} min={50} max={400} step={25} onChange={setEpochsTarget} />
            <Slider label="speed" value={speed} min={1} max={30} step={1} onChange={setSpeed} format={(v) => `${v} ep/frame`} />
            <Slider label="seed" value={seed} min={1} max={20} step={1} onChange={setSeed} />
            <p className="text-[11px] leading-relaxed text-zinc-500 dark:text-zinc-400">
              Same network <code>[2→{width}→{width}→1]</code> trained on 20, 200, and 2000 points from the same noisy
              distribution. Capacity is fixed; only the data grows. Watch the gap shrink left-to-right.
            </p>
          </div>
        </div>
      </div>
    </ClaimCard>
  )
}
