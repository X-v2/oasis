"use client";

import { useMemo } from "react";
import { ExtrudeGeometry, Shape } from "three";

import type { RoomRenderNode } from "@/lib/sceneGraph";

type SlabMeshProps = {
  node: RoomRenderNode;
  isSelected: boolean;
};

export default function SlabMesh({ node, isSelected }: SlabMeshProps) {
  const shape = useMemo(() => {
    const nextShape = new Shape();
    node.polygon2D.forEach(([x, z], index) => {
      if (index === 0) {
        nextShape.moveTo(x, z);
      } else {
        nextShape.lineTo(x, z);
      }
    });
    nextShape.closePath();
    return nextShape;
  }, [node]);

  const slabGeometry = useMemo(
    () =>
      new ExtrudeGeometry(shape, {
        depth: node.thickness,
        bevelEnabled: false,
        steps: 1,
      }),
    [node.thickness, shape],
  );
  const slabBaseElevation = 0;

  return (
    <mesh
      castShadow={false}
      receiveShadow
      geometry={slabGeometry}
      rotation={[Math.PI / 2, 0, 0]}
      position={[0, slabBaseElevation, 0]}
      userData={{ interactive: true, type: "slab", id: node.id }}
    >
      <meshStandardMaterial
        color={isSelected ? "#d8d0c2" : node.fillColor}
        roughness={0.96}
        metalness={0.02}
        polygonOffset
        polygonOffsetFactor={-1}
        polygonOffsetUnits={1}
      />
    </mesh>
  );
}
