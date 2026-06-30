import { Canvas } from '@react-three/fiber'
import { OrbitControls, Grid } from '@react-three/drei'
import { useMemo } from 'react'
import * as THREE from 'three'

// Generic 3D "feature lift" viewer.
//
// `points` is an array of { x, y, z, label }: the original 2D coordinates (x, y)
// plus a learned height z (the network's pre-sigmoid logit, supplied by the
// caller). The horizontal plane at height 0 is the network's decision surface
// (sigmoid = 0.5). When the lift separates the two rings, the plane slips
// cleanly between them; without a nonlinearity the lift is just a tilted plane
// and the rings stay interleaved. Drag to orbit.
function Cloud({ points }) {
  const { geometry } = useMemo(() => {
    const positions = new Float32Array(points.length * 3)
    const colors = new Float32Array(points.length * 3)
    const c0 = new THREE.Color('#ef4444')
    const c1 = new THREE.Color('#3b82f6')
    points.forEach((p, i) => {
      positions[i * 3] = p.x
      positions[i * 3 + 1] = p.z // three is y-up: vertical axis = learned height
      positions[i * 3 + 2] = p.y
      const c = p.label === 0 ? c0 : c1
      colors[i * 3] = c.r
      colors[i * 3 + 1] = c.g
      colors[i * 3 + 2] = c.b
    })
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    g.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    return { geometry: g }
  }, [points])

  return (
    <points geometry={geometry}>
      <pointsMaterial size={0.16} vertexColors sizeAttenuation />
    </points>
  )
}

export default function RingLift3D({ points }) {
  return (
    <div className="h-[340px] w-full overflow-hidden rounded-lg bg-zinc-900 ring-1 ring-zinc-700">
      <Canvas camera={{ position: [4.5, 4, 5.5], fov: 50 }} dpr={[1, 2]}>
        <color attach="background" args={['#0c1322']} />
        {/* Decision surface: the plane the network separates classes with. */}
        <mesh rotation={[-Math.PI / 2, 0, 0]}>
          <planeGeometry args={[8, 8]} />
          <meshBasicMaterial color="#64748b" transparent opacity={0.18} side={THREE.DoubleSide} />
        </mesh>
        <Grid
          args={[8, 8]}
          cellColor="#1e293b"
          sectionColor="#334155"
          fadeDistance={18}
          infiniteGrid={false}
          position={[0, 0.001, 0]}
        />
        <Cloud points={points} />
        <OrbitControls enablePan={false} minDistance={3} maxDistance={16} />
      </Canvas>
    </div>
  )
}
