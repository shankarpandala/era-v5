import { useCallback, useMemo, useState } from 'react'
import ClaimCard from '../components/ClaimCard.jsx'
import ScatterCanvas from '../components/ui/ScatterCanvas.jsx'
import Slider from '../components/ui/Slider.jsx'
import Button from '../components/ui/Button.jsx'
import useTrainer from '../hooks/useTrainer.js'
import { makeRng } from '../lib/rng.js'
import { makeRings } from '../lib/datasets.js'
import { MLP, Adam, matChain } from '../lib/nn.js'

const ACCENT = '#8b5cf6'
const WIDTH = 12 // width of the deep linear/ReLU stacks
const HIDDEN = [WIDTH, WIDTH, WIDTH, WIDTH] // 5 weight matrices: 2→12→12→12→12→1

function Acc({ label, acc }) {
  return (
    <div className="text-center">
      <div className="text-[10px] uppercase tracking-wide text-zinc-500">{label}</div>
      <div className="font-mono text-sm font-semibold text-zinc-800 dark:text-zinc-100">
        {acc == null ? '—' : `${(acc * 100).toFixed(1)}%`}
      </div>
    </div>
  )
}

// Shows the five weight matrices of the linear stack as shape badges, then the
// single 2×1 matrix they collapse to when multiplied — depth without
// nonlinearity is algebraically one linear map.
function MatrixCollapse({ collapsed }) {
  const shapes = [`2×${WIDTH}`, `${WIDTH}×${WIDTH}`, `${WIDTH}×${WIDTH}`, `${WIDTH}×${WIDTH}`, `${WIDTH}×1`]
  return (
    <div className="mt-4 rounded-lg bg-zinc-50 p-3 dark:bg-zinc-800/50">
      <div className="mb-2 text-xs font-medium text-zinc-600 dark:text-zinc-300">
        Bonus: multiply the five weight matrices of the 5-linear net
      </div>
      <div className="flex flex-wrap items-center gap-1.5 font-mono text-[11px]">
        {shapes.map((s, i) => (
          <span key={i} className="flex items-center gap-1.5">
            <span className="rounded bg-violet-100 px-1.5 py-0.5 text-violet-700 dark:bg-violet-900/40 dark:text-violet-300">
              W<sub>{i + 1}</sub> [{s}]
            </span>
            {i < shapes.length - 1 && <span className="text-zinc-400">·</span>}
          </span>
        ))}
        <span className="mx-1 text-zinc-500">=</span>
        <span className="rounded bg-emerald-100 px-2 py-1 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300">
          {collapsed ? (
            <>
              W<sub>eff</sub> [2×1] = [{collapsed[0].toFixed(3)}, {collapsed[1].toFixed(3)}]
            </>
          ) : (
            '…'
          )}
        </span>
      </div>
      <div className="mt-2 text-[11px] text-zinc-500 dark:text-zinc-400">
        Five weight matrices, one matrix. The deep linear net's map is exactly W<sub>eff</sub>·x + b — the same
        affine hypothesis class as a single layer, so it draws the same line. Depth bought nothing.
      </div>
    </div>
  )
}

export default function S1DepthLinear() {
  const [n, setN] = useState(300)
  const [noise, setNoise] = useState(0.18)
  const [lr, setLr] = useState(0.05)
  const [epochsTarget, setEpochsTarget] = useState(450)
  const [seed, setSeed] = useState(1)
  const [activated, setActivated] = useState(false)

  const data = useMemo(() => makeRings(n, noise, makeRng(seed)), [n, noise, seed])

  const build = useCallback(() => {
    const mk = (hidden, activation, s) =>
      new MLP({ inDim: 2, hidden, activation, outDim: 1, head: 'sigmoid', rng: makeRng(seed * 100 + s) })
    const one = mk([], 'none', 1)
    const fiveLin = mk(HIDDEN, 'none', 2)
    const fiveRelu = mk(HIDDEN, 'relu', 3)
    const opts = [new Adam({ lr }), new Adam({ lr }), new Adam({ lr })]
    const { X, y } = data
    return {
      one,
      fiveLin,
      fiveRelu,
      step() {
        one.trainStep(X, y, opts[0])
        fiveLin.trainStep(X, y, opts[1])
        fiveRelu.trainStep(X, y, opts[2])
      },
      snapshot() {
        return {
          accOne: one.accuracy(X, y),
          accLin: fiveLin.accuracy(X, y),
          accRelu: fiveRelu.accuracy(X, y),
        }
      },
    }
  }, [data, lr, seed])

  const { tick, running, snapshot, trainer, epoch, start, stop, reset } = useTrainer({
    build,
    epochs: epochsTarget,
    autoStart: activated,
    deps: [data, lr, seed, epochsTarget],
  })

  // Hold off training until the user engages (hover or click), so the untrained
  // "before" state is visible on load.
  const begin = () => {
    setActivated(true)
    start()
  }

  const collapsed = useMemo(() => {
    const t = trainer.current
    if (!t) return null
    const W = matChain(t.fiveLin.denseWeights()) // (2×1)
    return [W.d[0], W.d[1]]
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick])

  return (
    <ClaimCard
      id="s1-2"
      code="S1-2"
      accent={ACCENT}
      title="Depth without nonlinearity is a lie"
      claim="Five stacked linear layers collapse to a single linear map, so a 5-layer linear net is no stronger than 1 layer — both fail the ring task identically. Inserting ReLUs between the same five layers suddenly solves it."
      takeaway="The 1-layer and 5-linear boundaries are the same line at the same accuracy — and the five weight matrices literally multiply out to one 2×1 matrix. Depth alone is an illusion; the ReLU stack is the only one that bends the boundary around the ring."
    >
      <div className="panel p-4" onMouseEnter={() => !activated && begin()}>
        <div className="grid gap-6 lg:grid-cols-[1fr_260px]">
          <div>
            <div className="grid gap-4 sm:grid-cols-3">
              <div>
                <ScatterCanvas points={data.points} domain={data.domain} model={trainer.current?.one} redrawKey={tick} running={running} size={240} title="1 linear layer" />
                <div className="mt-2">
                  <Acc label="train acc" acc={snapshot?.accOne} />
                </div>
              </div>
              <div>
                <ScatterCanvas points={data.points} domain={data.domain} model={trainer.current?.fiveLin} redrawKey={tick} running={running} size={240} title="5 linear layers" />
                <div className="mt-2">
                  <Acc label="train acc" acc={snapshot?.accLin} />
                </div>
              </div>
              <div>
                <ScatterCanvas points={data.points} domain={data.domain} model={trainer.current?.fiveRelu} redrawKey={tick} running={running} size={240} title="5 layers + ReLU" />
                <div className="mt-2">
                  <Acc label="train acc" acc={snapshot?.accRelu} />
                </div>
              </div>
            </div>

            <MatrixCollapse collapsed={collapsed} />

            <div className="mt-4 flex flex-wrap items-center gap-3">
              <Button onClick={running ? stop : epoch >= epochsTarget ? () => reset(true) : begin}>
                {running ? '⏸ Pause' : epoch >= epochsTarget ? '↻ Re-train' : '▶ Train'}
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
            <Slider label="noise" value={noise} min={0} max={0.5} step={0.01} onChange={setNoise} format={(v) => v.toFixed(2)} />
            <Slider label="points" value={n} min={60} max={600} step={20} onChange={setN} />
            <Slider label="learning rate" value={lr} min={0.005} max={0.2} step={0.005} onChange={setLr} format={(v) => v.toFixed(3)} />
            <Slider label="epochs" value={epochsTarget} min={100} max={1000} step={50} onChange={setEpochsTarget} />
            <Slider label="seed" value={seed} min={1} max={20} step={1} onChange={setSeed} />
            <p className="text-[11px] leading-relaxed text-zinc-500 dark:text-zinc-400">
              Left two panels stay a straight line however long they train. Only the ReLU stack (right) wraps the
              ring. Watch W<sub>eff</sub> below update live as the linear stack trains.
            </p>
          </div>
        </div>
      </div>
    </ClaimCard>
  )
}
