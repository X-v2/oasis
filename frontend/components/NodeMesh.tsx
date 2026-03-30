"use client";

import { useMemo } from "react";
import { MeshBasicMaterial, MeshStandardMaterial, SphereGeometry } from "three";

import type { StructuralNodeRenderNode } from "@/lib/sceneGraph";

type NodeMeshProps = {
  node: StructuralNodeRenderNode;
  isSelected: boolean;
};

export default function NodeMesh({ node, isSelected }: NodeMeshProps) {
  const shellGeometry = useMemo(() => new SphereGeometry(isSelected ? 0.106 : 0.096, 22, 22), [isSelected]);
  const coreGeometry = useMemo(() => new SphereGeometry(isSelected ? 0.074 : 0.066, 18, 18), [isSelected]);
  const hitRadius = node.degree >= 4 ? 0.26 : node.likelyColumn ? 0.23 : 0.2;
  const hitGeometry = useMemo(() => new SphereGeometry(hitRadius, 16, 16), [hitRadius]);

  const shellMaterial = useMemo(
    () =>
      new MeshStandardMaterial({
        color: "#2d4050",
        roughness: 0.68,
        metalness: 0.1,
        emissive: "#111b23",
        emissiveIntensity: 0.08,
        depthTest: false,
        depthWrite: false,
        fog: false,
      }),
    [],
  );
  const coreMaterial = useMemo(
    () =>
      new MeshStandardMaterial({
        color: isSelected ? "#577188" : node.color,
        roughness: 0.54,
        metalness: 0.14,
        emissive: isSelected ? "#22303c" : "#17222b",
        emissiveIntensity: isSelected ? 0.24 : 0.14,
        depthTest: false,
        depthWrite: false,
        fog: false,
      }),
    [isSelected, node.color],
  );
  const hitMaterial = useMemo(
    () =>
      new MeshBasicMaterial({
        transparent: true,
        opacity: 0,
        depthWrite: false,
        depthTest: false,
      }),
    [],
  );

  return (
    <group position={[node.position[0], node.position[1] + 0.022, node.position[2]]} renderOrder={12}>
      <mesh castShadow={false} receiveShadow={false} geometry={shellGeometry} material={shellMaterial} frustumCulled={false} />
      <mesh castShadow={false} receiveShadow={false} geometry={coreGeometry} material={coreMaterial} frustumCulled={false} />
      <mesh
        castShadow={false}
        receiveShadow={false}
        geometry={hitGeometry}
        material={hitMaterial}
        userData={{ interactive: true, type: "node", id: node.id }}
        frustumCulled={false}
      />
    </group>
  );
}
