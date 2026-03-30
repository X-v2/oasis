"use client";

import { useMemo } from "react";
import { BoxGeometry } from "three";

import type { WallRenderNode } from "@/lib/sceneGraph";

type WallMeshProps = {
  node: WallRenderNode;
  isSelected: boolean;
};

export default function WallMesh({ node, isSelected }: WallMeshProps) {
  const bodyGeometry = useMemo(() => new BoxGeometry(node.length, node.height, node.thickness), [node.height, node.length, node.thickness]);
  const hitGeometry = useMemo(
    () => new BoxGeometry(node.length + 0.08, Math.max(node.height + 0.12, 0.35), Math.max(node.thickness + 0.26, 0.36)),
    [node.height, node.length, node.thickness],
  );

  return (
    <group position={node.position} rotation={[0, -node.wall.angle, 0]}>
      <mesh castShadow receiveShadow geometry={bodyGeometry} userData={{ interactive: true, type: "wall", id: node.wall.id }}>
        <meshStandardMaterial
          color={isSelected ? "#b6ab98" : node.bodyColor}
          metalness={0.02}
          roughness={0.9}
          polygonOffset
          polygonOffsetFactor={1}
          polygonOffsetUnits={1}
        />
      </mesh>
      <mesh geometry={hitGeometry} userData={{ interactive: true, type: "wall", id: node.wall.id }}>
        <meshBasicMaterial transparent opacity={0} depthWrite={false} />
      </mesh>
    </group>
  );
}
