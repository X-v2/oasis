"use client";

import { useMemo } from "react";
import { BoxGeometry } from "three";

import type { ColumnRenderNode } from "@/lib/sceneGraph";

type ColumnMeshProps = {
  node: ColumnRenderNode;
  isSelected: boolean;
};

export default function ColumnMesh({ node, isSelected }: ColumnMeshProps) {
  const geometry = useMemo(
    () =>
      new BoxGeometry(
        isSelected ? node.width + 0.05 : node.width,
        node.height,
        isSelected ? node.depth + 0.05 : node.depth,
      ),
    [isSelected, node.depth, node.height, node.width],
  );

  return (
    <mesh
      castShadow
      receiveShadow
      geometry={geometry}
      position={node.position}
      renderOrder={5}
      userData={{ interactive: true, type: "column", id: node.id, nodeId: node.nodeId }}
    >
      <meshStandardMaterial color={isSelected ? "#8b7e68" : node.color} metalness={0.08} roughness={0.82} />
    </mesh>
  );
}
