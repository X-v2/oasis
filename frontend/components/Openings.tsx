"use client";

import { useMemo, useRef, useState } from "react";
import { useFrame } from "@react-three/fiber";
import { BoxGeometry, DoubleSide, type Group } from "three";

import type { OpeningRenderNode } from "@/lib/sceneGraph";

type OpeningsProps = {
  nodes: OpeningRenderNode[];
};

// Critically damped smooth interpolation — eliminates shivering/jitter
function smoothDamp(current: number, target: number, factor: number) {
  const delta = target - current;
  if (Math.abs(delta) < 0.0001) return target;
  return current + delta * factor;
}

// ── Door leaf ─────────────────────────────────────────────────────────────
// Reduced max swing from 0.55π → 0.42π (≈75°) for a natural, non-theatrical open
// Hollow opening rule: when open, the door panel swings away revealing empty space.
// The wall boarding around the opening is NOT rendered here — the geometry gap
// in the WallMesh already creates the hollow. The panel simply moves aside.
function DoorLeaf({ node }: { node: OpeningRenderNode }) {
  const [open, setOpen] = useState(false);
  const swingRef = useRef<Group>(null);
  const currentAngleRef = useRef(0);
  const swingRight = node.opening.swing === "right";
  const leafOffset = swingRight ? -node.opening.width / 2 : node.opening.width / 2;

  // Reduced rotation: 75° feels natural without being theatrical
  const maxSwingAngle = Math.PI * 0.42;
  const targetAngle = open ? (swingRight ? maxSwingAngle : -maxSwingAngle) : 0;

  const leafGeometry = useMemo(
    () => new BoxGeometry(node.opening.width, node.opening.height, 0.042),
    [node.opening.height, node.opening.width],
  );

  useFrame((_, delta) => {
    if (!swingRef.current) return;
    // Time-based critically-damped easing — no jitter, consistent speed
    const lerpFactor = 1 - Math.exp(-8 * delta);
    currentAngleRef.current = smoothDamp(currentAngleRef.current, targetAngle, lerpFactor);
    swingRef.current.rotation.y = currentAngleRef.current;
  });

  function toggleDoor(event: { stopPropagation: () => void }) {
    event.stopPropagation();
    setOpen((current) => !current);
  }

  return (
    <group onClick={toggleDoor}>
      {/* Swinging door leaf */}
      <group position={node.hinge3D} rotation={[0, -node.angle, 0]}>
        <group ref={swingRef}>
          {/* Main door panel */}
          <mesh
            position={[leafOffset, node.opening.height / 2, 0.022]}
            castShadow={false}
            receiveShadow={false}
            geometry={leafGeometry}
            renderOrder={4}
            userData={{ interactive: true, type: "door", id: node.id }}
          >
            <meshStandardMaterial color="#6b5e52" roughness={0.72} metalness={0.05} />
          </mesh>
          {/* Door panel inset detail */}
          <mesh
            position={[leafOffset, node.opening.height / 2, 0.045]}
            castShadow={false}
            receiveShadow={false}
          >
            <boxGeometry args={[node.opening.width * 0.7, node.opening.height * 0.55, 0.006]} />
            <meshStandardMaterial color="#5a5048" roughness={0.85} metalness={0} />
          </mesh>
          {/* Door handle */}
          <mesh
            position={[swingRight ? leafOffset + node.opening.width * 0.38 : leafOffset - node.opening.width * 0.38, node.opening.height * 0.48, 0.065]}
            castShadow={false}
            receiveShadow={false}
          >
            <sphereGeometry args={[0.028, 8, 8]} />
            <meshStandardMaterial color="#a09080" roughness={0.3} metalness={0.6} />
          </mesh>
        </group>
      </group>
    </group>
  );
}

// ── Window panel ──────────────────────────────────────────────────────────
// When open: glass panels swing away, revealing hollow (transparent) opening.
// Smooth damped animation, no jitter, reduced max swing to 60°.
function WindowPanel({ node }: { node: OpeningRenderNode }) {
  const [open, setOpen] = useState(false);
  const isSinglePanel = node.opening.panelType === "single";
  const singlePanelRef = useRef<Group>(null);
  const leftPanelRef = useRef<Group>(null);
  const rightPanelRef = useRef<Group>(null);

  const singleCurrentRef = useRef(0);
  const leftCurrentRef = useRef(0);
  const rightCurrentRef = useRef(0);

  // Reduced max swing: 60° (π/3) — panels open fully without over-rotation
  const maxSwingAngle = Math.PI / 3;
  const singleTarget = open ? maxSwingAngle : 0;
  const leftTarget = open ? maxSwingAngle : 0;
  const rightTarget = open ? -maxSwingAngle : 0;

  const faceOffset = Math.max(0.018, Math.min(0.032, node.wall.thickness * 0.16));
  const singleGlassGeometry = useMemo(
    () => new BoxGeometry(node.opening.width, node.opening.height, 0.006),
    [node.opening.height, node.opening.width],
  );
  const splitGlassGeometry = useMemo(
    () => new BoxGeometry(node.opening.width / 2, node.opening.height, 0.006),
    [node.opening.height, node.opening.width],
  );

  useFrame((_, delta) => {
    const lerpFactor = 1 - Math.exp(-7 * delta);

    if (singlePanelRef.current) {
      singleCurrentRef.current = smoothDamp(singleCurrentRef.current, singleTarget, lerpFactor);
      singlePanelRef.current.rotation.y = singleCurrentRef.current;
    }
    if (leftPanelRef.current) {
      leftCurrentRef.current = smoothDamp(leftCurrentRef.current, leftTarget, lerpFactor);
      leftPanelRef.current.rotation.y = leftCurrentRef.current;
    }
    if (rightPanelRef.current) {
      rightCurrentRef.current = smoothDamp(rightCurrentRef.current, rightTarget, lerpFactor);
      rightPanelRef.current.rotation.y = rightCurrentRef.current;
    }
  });

  // Glass material properties — more realistic, less opaque
  const glassMaterialProps = {
    color: "#bfe7ff" as const,
    transparent: true as const,
    opacity: open ? 0 : 0.3, // becomes fully transparent/hollow when open
    transmission: 0.74,
    thickness: 0.004,
    roughness: 0.08,
    metalness: 0,
    ior: 1.52,
    side: DoubleSide,
    polygonOffset: true as const,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  };

  function toggleWindow(event: { stopPropagation: () => void }) {
    event.stopPropagation();
    setOpen((current) => !current);
  }

  return (
    <group onClick={toggleWindow}>
      {/* Glass panes — swing open, become invisible (hollow) when fully open */}
      {isSinglePanel ? (
        <group position={node.hingeLeft3D} rotation={[0, -node.angle, 0]}>
          <group ref={singlePanelRef}>
            <mesh
              castShadow={false}
              receiveShadow={false}
              geometry={singleGlassGeometry}
              position={[node.opening.width / 2, node.opening.height / 2, faceOffset]}
              renderOrder={4}
              userData={{ interactive: true, type: "window", id: node.id }}
            >
              <meshPhysicalMaterial
                ref={null}
                {...glassMaterialProps}
              />
            </mesh>
          </group>
        </group>
      ) : (
        <>
          <group position={node.hingeLeft3D} rotation={[0, -node.angle, 0]}>
            <group ref={leftPanelRef}>
              <mesh
                castShadow={false}
                receiveShadow={false}
                geometry={splitGlassGeometry}
                position={[node.opening.width / 4, node.opening.height / 2, faceOffset]}
                renderOrder={4}
                userData={{ interactive: true, type: "window", id: node.id }}
              >
                <meshPhysicalMaterial
                  ref={null}
                  {...glassMaterialProps}
                />
              </mesh>
            </group>
          </group>
          <group position={node.hingeRight3D} rotation={[0, -node.angle, 0]}>
            <group ref={rightPanelRef}>
              <mesh
                castShadow={false}
                receiveShadow={false}
                geometry={splitGlassGeometry}
                position={[-node.opening.width / 4, node.opening.height / 2, faceOffset]}
                renderOrder={4}
                userData={{ interactive: true, type: "window", id: node.id }}
              >
                <meshPhysicalMaterial
                  ref={null}
                  {...glassMaterialProps}
                />
              </mesh>
            </group>
          </group>
        </>
      )}
    </group>
  );
}

export default function Openings({ nodes }: OpeningsProps) {
  return (
    <>
      {nodes.map((node) =>
        node.opening.kind === "door" ? <DoorLeaf key={node.id} node={node} /> : <WindowPanel key={node.id} node={node} />,
      )}
    </>
  );
}
