import { lazy, Suspense } from 'react'

// Dispatch a mechanism's `viz: { kind, mode }` to one of the shared widgets.
// Widgets are lazy so the timeline (70+ cards) only pays for the ones opened.
const AttentionMatrix = lazy(() => import('./AttentionMatrix.jsx'))
const SoftmaxPipeline = lazy(() => import('./SoftmaxPipeline.jsx'))
const PositionExplorer = lazy(() => import('./PositionExplorer.jsx'))
const KVCacheHeads = lazy(() => import('./KVCacheHeads.jsx'))
const RecurrentState = lazy(() => import('./RecurrentState.jsx'))

function Fallback() {
  return <div className="panel h-40 animate-pulse p-4 text-xs text-zinc-400">loading visualizer…</div>
}

export default function Viz({ viz }) {
  if (!viz || viz.kind === 'none') return null
  let el = null
  switch (viz.kind) {
    case 'baseline':
      el = <SoftmaxPipeline />
      break
    case 'matrix':
      el = <AttentionMatrix mode={viz.mode} params={viz.params} />
      break
    case 'position':
      el = <PositionExplorer mode={viz.mode} />
      break
    case 'kv':
      el = <KVCacheHeads mode={viz.mode} />
      break
    case 'state':
      el = <RecurrentState mode={viz.mode} />
      break
    default:
      return null
  }
  return <Suspense fallback={<Fallback />}>{el}</Suspense>
}
