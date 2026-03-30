"use client";

import { useEffect, useMemo, useState } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { AxesHelper, PCFShadowMap, Vector2, Vector3 } from "three";
import { Html, Line, OrbitControls, Text } from "@react-three/drei";

import { classifyWall, getRoomContextForWall } from "@/lib/materialEngine";
import { buildSceneGraph } from "@/lib/sceneGraph";
import type { OptimizationPreviewLine, SceneData, SceneSelection, SceneSelectionType, SpanPreviewLine } from "@/lib/types";

import ColumnMesh from "./ColumnMesh";
import NodeMesh from "./NodeMesh";
import Openings from "./Openings";
import SlabMesh from "./SlabMesh";
import WallMesh from "./WallMesh";

type Viewer3DProps = {
  data: SceneData;
  selectedEntity: SceneSelection | null;
  structuralView: boolean;
  debugOverlay: boolean;
  darkMode: boolean;
  onSelectEntity: (selection: SceneSelection | null) => void;
  focusPoint: [number, number, number] | null;
  focusToken: number;
  spanPreviewLines?: SpanPreviewLine[];
  optimizationPreviewLines?: OptimizationPreviewLine[];
};

type HoverPreview = {
  type: SceneSelectionType;
  id: string;
  x: number;
  y: number;
} | null;

type PickMode = "auto" | "walls" | "nodes";

function CameraRig({ data }: { data: SceneData }) {
  const { camera } = useThree();
  const { center, width, depth } = data.bounds;

  useEffect(() => {
    const maxDimension = Math.max(width, depth, 8);
    camera.position.set(
      center[0] + maxDimension * 0.95,
      Math.max(8, maxDimension * 0.82),
      center[2] + maxDimension * 0.88,
    );
    camera.lookAt(center[0], 0.8, center[2]);
    camera.updateProjectionMatrix();
  }, [camera, center[0], center[2], depth, width]);

  return null;
}

function Axes({ debugOverlay }: { debugOverlay: boolean }) {
  const { scene } = useThree();

  useEffect(() => {
    if (!debugOverlay) {
      return undefined;
    }

    const helper = new AxesHelper(2.5);
    helper.position.set(0, 0.02, 0);
    scene.add(helper);

    return () => {
      scene.remove(helper);
    };
  }, [debugOverlay, scene]);

  return null;
}

function FocusGuide({
  focusPoint,
  focusToken,
  bounds,
}: {
  focusPoint: [number, number, number] | null;
  focusToken: number;
  bounds: SceneData["bounds"];
}) {
  const { camera, controls } = useThree();

  useEffect(() => {
    if (!focusPoint) {
      return;
    }

    const target = new Vector3(focusPoint[0], Math.max(0.2, focusPoint[1]), focusPoint[2]);
    const maxDimension = Math.max(bounds.width, bounds.depth, 8);
    const nextPosition = new Vector3(
      target.x + maxDimension * 0.52,
      target.y + Math.max(2.6, maxDimension * 0.34),
      target.z + maxDimension * 0.52,
    );

    camera.position.copy(nextPosition);
    camera.lookAt(target);
    camera.updateProjectionMatrix();

    const orbit = controls as { target?: Vector3; update?: () => void } | undefined;
    if (orbit?.target) {
      orbit.target.copy(target);
      orbit.update?.();
    }
  }, [bounds.depth, bounds.width, camera, controls, focusPoint, focusToken]);

  return null;
}

function InteractionLayer({
  onSelectEntity,
  onHoverPreview,
  active,
  pickMode,
}: {
  onSelectEntity: (selection: SceneSelection | null) => void;
  onHoverPreview: (preview: HoverPreview) => void;
  active: boolean;
  pickMode: PickMode;
}) {
  const { camera, gl, raycaster, scene } = useThree();

  useEffect(() => {
    const priority: SceneSelectionType[] = ["door", "window", "node", "wall", "column", "slab"];
    const hitPriorityByType: Record<SceneSelectionType, number> = {
      door: 0,
      window: 1,
      node: 2,
      wall: 3,
      column: 4,
      slab: 5,
    };
    const pointer = new Vector2();
    let pointerDownPosition: { x: number; y: number } | null = null;
    let lastPickSignature = "";
    let lastPickPoint: { x: number; y: number } | null = null;
    let lastPickIndex = 0;
    let lastHoverSignature = "";

    function allowedByPickMode(type: SceneSelectionType) {
      if (pickMode === "auto") {
        return true;
      }
      if (pickMode === "walls") {
        return type === "wall";
      }
      return type === "node" || type === "column";
    }

    function selectionFromUserData(userData: Record<string, unknown>) {
      const data = userData as Partial<SceneSelection> & { nodeId?: string };
      if (!data.type || !data.id) {
        return null;
      }
      if (data.type === "column" && data.nodeId) {
        return { type: "node" as const, id: data.nodeId };
      }
      return { type: data.type, id: data.id };
    }

    function pickHit(event: MouseEvent) {
      if (!active) {
        return { candidates: [], bounds: gl.domElement.getBoundingClientRect() };
      }
      const bounds = gl.domElement.getBoundingClientRect();
      pointer.set(
        ((event.clientX - bounds.left) / bounds.width) * 2 - 1,
        -((event.clientY - bounds.top) / bounds.height) * 2 + 1,
      );

      raycaster.setFromCamera(pointer, camera);
      const candidates = raycaster
        .intersectObjects(scene.children, true)
        .filter((intersection) => {
          const data = intersection.object.userData as Partial<SceneSelection> & { interactive?: boolean };
          const type = data.type as SceneSelectionType | undefined;
          return Boolean(data.interactive && type && data.id && allowedByPickMode(type));
        })
        .sort((a, b) => {
          const typeA = (a.object.userData.type as SceneSelectionType | undefined) ?? "wall";
          const typeB = (b.object.userData.type as SceneSelectionType | undefined) ?? "wall";
          const rankA = hitPriorityByType[typeA] ?? priority.indexOf(typeA);
          const rankB = hitPriorityByType[typeB] ?? priority.indexOf(typeB);
          const priorityDelta = rankA - rankB;

          return priorityDelta || a.distance - b.distance;
        })
        .filter((hit, index, all) => {
          const current = selectionFromUserData(hit.object.userData as Record<string, unknown>);
          if (!current) {
            return false;
          }
          for (let pointerIndex = 0; pointerIndex < index; pointerIndex += 1) {
            const previous = selectionFromUserData(all[pointerIndex].object.userData as Record<string, unknown>);
            if (previous?.type === current.type && previous.id === current.id) {
              return false;
            }
          }
          return true;
        });

      return { candidates, bounds };
    }

    function handlePointerDown(event: PointerEvent) {
      if (!active) {
        return;
      }
      pointerDownPosition = { x: event.clientX, y: event.clientY };
    }

    function handlePointerUp(event: PointerEvent) {
      if (!active) {
        return;
      }
      if (!pointerDownPosition) {
        return;
      }

      const moveDistance = Math.hypot(event.clientX - pointerDownPosition.x, event.clientY - pointerDownPosition.y);
      pointerDownPosition = null;
      if (moveDistance > 4) {
        return;
      }

      const { candidates } = pickHit(event);
      if (candidates.length === 0) {
        lastPickSignature = "";
        lastPickPoint = null;
        lastPickIndex = 0;
        onSelectEntity(null);
        return;
      }

      const signature = candidates
        .map((candidate) => {
          const selection = selectionFromUserData(candidate.object.userData as Record<string, unknown>);
          return selection ? `${selection.type}:${selection.id}` : "";
        })
        .filter(Boolean)
        .join("|");
      const repeatedLocation = lastPickPoint
        ? Math.hypot(event.clientX - lastPickPoint.x, event.clientY - lastPickPoint.y) < 7
        : false;
      const sameStack = signature.length > 0 && signature === lastPickSignature && repeatedLocation;

      lastPickIndex = sameStack ? (lastPickIndex + 1) % candidates.length : 0;
      lastPickSignature = signature;
      lastPickPoint = { x: event.clientX, y: event.clientY };

      const chosen = candidates[lastPickIndex];
      const chosenSelection = selectionFromUserData(chosen.object.userData as Record<string, unknown>);
      if (!chosenSelection) {
        return;
      }
      onSelectEntity(chosenSelection);
    }

    function handleMove(event: MouseEvent) {
      if (!active) {
        if (lastHoverSignature !== "none") {
          lastHoverSignature = "none";
          onHoverPreview(null);
        }
        return;
      }
      const { candidates, bounds } = pickHit(event);
      if (candidates.length === 0) {
        if (lastHoverSignature !== "none") {
          lastHoverSignature = "none";
          onHoverPreview(null);
        }
        return;
      }

      const topHitData = candidates[0].object.userData as SceneSelection & { nodeId?: string };
      const previewType = topHitData.type === "column" && topHitData.nodeId ? "node" : topHitData.type;
      const previewId = topHitData.type === "column" && topHitData.nodeId ? topHitData.nodeId : topHitData.id;
      const overlapCount = Math.max(0, candidates.length - 1);
      const composedId = overlapCount > 0 ? `${previewId} (+${overlapCount})` : previewId;
      const signature = `${previewType}:${composedId}`;
      if (signature === lastHoverSignature) {
        return;
      }
      lastHoverSignature = signature;
      onHoverPreview({
        type: previewType,
        id: composedId,
        x: event.clientX - bounds.left,
        y: event.clientY - bounds.top,
      });
    }

    function handleLeave() {
      pointerDownPosition = null;
      lastPickSignature = "";
      lastPickPoint = null;
      lastPickIndex = 0;
      lastHoverSignature = "";
      onHoverPreview(null);
    }

    gl.domElement.addEventListener("pointerdown", handlePointerDown);
    gl.domElement.addEventListener("pointerup", handlePointerUp);
    gl.domElement.addEventListener("mousemove", handleMove);
    gl.domElement.addEventListener("mouseleave", handleLeave);

    return () => {
      gl.domElement.removeEventListener("pointerdown", handlePointerDown);
      gl.domElement.removeEventListener("pointerup", handlePointerUp);
      gl.domElement.removeEventListener("mousemove", handleMove);
      gl.domElement.removeEventListener("mouseleave", handleLeave);
    };
  }, [active, camera, gl, onHoverPreview, onSelectEntity, pickMode, raycaster, scene]);

  return null;
}

function SceneContent(props: Viewer3DProps) {
  const { data, selectedEntity, structuralView, debugOverlay, darkMode } = props;
  const selectedWallId = selectedEntity?.type === "wall" ? selectedEntity.id : null;
  const selectedSlabId = selectedEntity?.type === "slab" ? selectedEntity.id : null;
  const selectedWall = data.walls.find((wall) => wall.id === selectedWallId);
  const selectedContext = selectedWall ? getRoomContextForWall(selectedWall, data) : undefined;
  const graph = useMemo(() => buildSceneGraph(data, structuralView), [data, structuralView]);
  const anchorNodes = graph.nodes;

  return (
    <>
      <color attach="background" args={[darkMode ? "#0e0c16" : "#edeae3"]} />
      {/* Stable, non-flickering lighting setup */}
      <ambientLight intensity={darkMode ? 0.55 : 0.48} />
      <directionalLight
        castShadow
        intensity={darkMode ? 0.62 : 0.78}
        position={[12, 18, 8]}
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
        shadow-bias={-0.0003}
        shadow-normalBias={0.06}
        shadow-camera-left={-22}
        shadow-camera-right={22}
        shadow-camera-top={22}
        shadow-camera-bottom={-22}
        shadow-camera-near={1}
        shadow-camera-far={50}
      />
      {/* Fill light from opposite side to reduce harsh shadows */}
      <directionalLight
        intensity={darkMode ? 0.18 : 0.22}
        position={[-8, 10, -6]}
        castShadow={false}
      />
      <hemisphereLight args={darkMode ? ["#8b7bc4", "#0c0a18", 0.35] : ["#f0f0ff", "#8fa0b0", 0.28]} />
      {debugOverlay && (
        <gridHelper
          args={[Math.max(data.bounds.width, data.bounds.depth) + 8, 24, darkMode ? "#5b4f82" : "#9aa8b5", darkMode ? "#382f54" : "#cfd7de"]}
          position={[data.bounds.center[0], -0.001, data.bounds.center[2]]}
        />
      )}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[data.bounds.center[0], -0.16, data.bounds.center[2]]} receiveShadow>
        <planeGeometry args={[data.bounds.width + 16, data.bounds.depth + 16]} />
        <meshStandardMaterial color={darkMode ? "#14101e" : "#e4e0d8"} roughness={0.95} metalness={0} />
      </mesh>

      {graph.rooms.map((node) => (
        <SlabMesh key={node.id} node={node} isSelected={selectedSlabId === node.id} />
      ))}

      {graph.columns.map((node) => (
        <group key={node.id}>
          <ColumnMesh node={node} isSelected={selectedEntity?.type === "column" && selectedEntity.id === node.id} />
          {(debugOverlay || (selectedEntity?.type === "column" && selectedEntity.id === node.id)) && (
            <Text
              position={[node.position[0], node.height + 0.2, node.position[2]]}
              fontSize={0.13}
              color={darkMode ? "#d7caf5" : "#2d4659"}
              anchorX="center"
              anchorY="middle"
              outlineWidth={0.035}
              outlineColor="#f6f4ef"
              renderOrder={18}
              material-depthTest={false}
              material-depthWrite={false}
              frustumCulled={false}
            >
              {node.id}
            </Text>
          )}
        </group>
      ))}

      {graph.walls.map((node) => (
        <WallMesh key={node.id} node={node} isSelected={node.wall.id === selectedWallId} />
      ))}

      <Openings nodes={graph.openings} />

      {anchorNodes.map((node) => (
        <group key={node.id}>
          <NodeMesh node={node} isSelected={selectedEntity?.type === "node" && selectedEntity.id === node.id} />
          {(debugOverlay || (selectedEntity?.type === "node" && selectedEntity.id === node.id)) && (
            <Text
              position={[node.position[0], node.position[1] + 0.2, node.position[2]]}
              fontSize={0.11}
              color={darkMode ? "#d7caf5" : "#2f4657"}
              anchorX="center"
              anchorY="middle"
              outlineWidth={0.04}
              outlineColor="#f6f4ef"
              renderOrder={18}
              material-depthTest={false}
              material-depthWrite={false}
              frustumCulled={false}
            >
              {node.id}
            </Text>
          )}
        </group>
      ))}

      {data.labels.map((label) => (
        <Text
          key={`label-name-${label.id}`}
          position={label.position}
          fontSize={0.28}
          color={darkMode ? "#f6edcb" : "#244158"}
          anchorX="center"
          anchorY="middle"
          rotation={[-Math.PI / 2, 0, 0]}
          outlineWidth={0.045}
          outlineColor={darkMode ? "#111018" : "#f8f5ee"}
          letterSpacing={0.02}
          material-depthTest={false}
          material-depthWrite={false}
          renderOrder={16}
          frustumCulled={false}
        >
          {label.text}
        </Text>
      ))}

      {debugOverlay &&
        data.walls.map((wall) => (
          <Text
            key={`label-${wall.id}`}
            position={[wall.midpoint[0], wall.height + 0.28, wall.midpoint[2]]}
            fontSize={0.2}
            color={darkMode ? "#d7caf5" : "#172554"}
            anchorX="center"
            anchorY="middle"
            outlineWidth={0.03}
            outlineColor="#f6f4ef"
            material-depthTest={false}
            material-depthWrite={false}
            renderOrder={18}
            frustumCulled={false}
          >
            {`${wall.id} | ${classifyWall(wall)} | ${wall.confidence}`}
          </Text>
        ))}

      {debugOverlay &&
        data.rooms.map((room) => (
          <Text
            key={`room-debug-${room.id}`}
            position={[room.centroid[0], 0.12, room.centroid[2]]}
            fontSize={0.16}
            color={darkMode ? "#c9bde7" : "#374151"}
            anchorX="center"
            anchorY="middle"
            rotation={[-Math.PI / 2, 0, 0]}
            outlineWidth={0.03}
            outlineColor="#f6f4ef"
            material-depthTest={false}
            material-depthWrite={false}
            renderOrder={18}
            frustumCulled={false}
          >
            {`${room.id} | span ${room.span.toFixed(2)}m | ${room.confidence}`}
          </Text>
        ))}

      {debugOverlay && selectedWall && selectedContext?.room && (
        <>
          <Line
            points={
              selectedContext.spanLine
                ? [new Vector3(...selectedContext.spanLine[0]), new Vector3(...selectedContext.spanLine[1])]
                : [
                    new Vector3(selectedContext.room.centroid[0], 0.12, selectedContext.room.centroid[2]),
                    new Vector3(selectedWall.midpoint[0], selectedWall.midpoint[1], selectedWall.midpoint[2]),
                  ]
            }
            color="#4b5563"
            lineWidth={2.5}
          />
          <Html position={[selectedWall.midpoint[0], selectedWall.height + 0.76, selectedWall.midpoint[2]]}>
            <div className="rounded-full border border-stone-300 bg-white/90 px-3 py-1 text-xs font-semibold text-stone-800 shadow-sm">
              {`Span ${selectedContext.span.toFixed(2)}m | ${selectedContext.confidence}`}
            </div>
          </Html>
        </>
      )}

      {(props.spanPreviewLines ?? []).map((preview) => (
        <Line
          key={`span-preview-${preview.id}`}
          points={[new Vector3(...preview.from), new Vector3(...preview.to)]}
          color={preview.confidence === "low" ? "#f59e0b" : "#22d3ee"}
          lineWidth={2.5}
          dashed
          dashSize={0.28}
          gapSize={0.16}
          renderOrder={17}
        />
      ))}

      {(props.optimizationPreviewLines ?? []).map((preview) => (
        <Line
          key={`opt-preview-${preview.id}`}
          points={[new Vector3(...preview.from), new Vector3(...preview.to)]}
          color={preview.color}
          lineWidth={2.2}
          dashed
          dashSize={0.22}
          gapSize={0.14}
          renderOrder={17}
        />
      ))}
    </>
  );
}

export default function Viewer3D(props: Viewer3DProps) {
  const [hoverPreview, setHoverPreview] = useState<HoverPreview>(null);
  const [interactionEnabled, setInteractionEnabled] = useState(false);
  const [pickMode, setPickMode] = useState<PickMode>("auto");
  const orbitTarget = useMemo(
    () => new Vector3(props.data.bounds.center[0], 0.75, props.data.bounds.center[2]),
    [props.data.bounds.center[0], props.data.bounds.center[2]],
  );

  function activateInteraction() {
    if (!interactionEnabled) {
      setInteractionEnabled(true);
    }
  }

  return (
    <div
      className={`relative overflow-hidden rounded-[20px] border ${
        props.darkMode
          ? "border-white/[0.06] bg-[#0e0c16] shadow-[0_20px_60px_-20px_rgba(0,0,0,0.8)]"
          : "border-black/[0.08] bg-[#edeae3] shadow-[0_20px_60px_-20px_rgba(0,0,0,0.18)]"
      }`}
      style={{ height: "clamp(380px, calc(100vh - 14rem), 720px)" }}
      onPointerDown={activateInteraction}
    >
      <Canvas
        shadows={{ type: PCFShadowMap }}
        camera={{ fov: 42, near: 0.4, far: 140, position: [12, 10, 12] }}
        dpr={[1, 1.5]}
        gl={{ antialias: true, powerPreference: "high-performance", alpha: false, preserveDrawingBuffer: false }}
        frameloop="demand"
      >
        <InteractionLayer
          onSelectEntity={props.onSelectEntity}
          onHoverPreview={setHoverPreview}
          active={interactionEnabled}
          pickMode={pickMode}
        />
        <CameraRig data={props.data} />
        <FocusGuide focusPoint={props.focusPoint} focusToken={props.focusToken} bounds={props.data.bounds} />
        <Axes debugOverlay={props.debugOverlay} />
        <SceneContent {...props} />
        <OrbitControls
          makeDefault
          enabled={interactionEnabled}
          enableDamping
          dampingFactor={0.08}
          rotateSpeed={0.7}
          zoomSpeed={0.9}
          minPolarAngle={0.25}
          maxPolarAngle={Math.PI / 2.08}
          minDistance={2}
          maxDistance={80}
          target={orbitTarget}
        />
      </Canvas>
      {!interactionEnabled && (
        <div
          className="pointer-events-none absolute inset-0 flex items-center justify-center"
          style={{ background: props.darkMode ? "rgba(14,12,22,0.32)" : "rgba(255,255,255,0.22)" }}
        >
          <div style={{
            borderRadius: 10,
            border: `1px solid ${props.darkMode ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}`,
            background: props.darkMode ? "rgba(20,16,32,0.88)" : "rgba(255,255,255,0.88)",
            backdropFilter: "blur(8px)",
            padding: "8px 16px",
            fontSize: 11,
            fontWeight: 600,
            letterSpacing: "0.14em",
            textTransform: "uppercase" as const,
            color: props.darkMode ? "rgba(220,210,255,0.9)" : "rgba(60,55,50,0.9)",
          }}>
            Click to enable 3D controls
          </div>
        </div>
      )}
      {props.data.walls.length === 0 && (
        <div
          className="pointer-events-none absolute inset-0 flex items-center justify-center"
          style={{ backdropFilter: "blur(2px)", background: props.darkMode ? "rgba(14,12,22,0.5)" : "rgba(240,238,232,0.4)" }}
        >
          <div style={{
            borderRadius: 18,
            border: `1px solid ${props.darkMode ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)"}`,
            background: props.darkMode ? "rgba(18,14,28,0.92)" : "rgba(255,254,250,0.94)",
            backdropFilter: "blur(16px)",
            padding: "24px 32px",
            textAlign: "center" as const,
            maxWidth: 360,
          }}>
            <div style={{ width: 40, height: 40, margin: "0 auto 12px", borderRadius: 10, border: `1px solid ${props.darkMode ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.08)"}`, display: "flex", alignItems: "center", justifyContent: "center", background: props.darkMode ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.04)" }}>
              <svg width="18" height="18" fill="none" stroke={props.darkMode ? "rgba(200,190,230,0.7)" : "rgba(80,75,70,0.7)"} strokeWidth="1.5" viewBox="0 0 24 24">
                <path d="M21 7.5l-9-5.25L3 7.5m18 0l-9 5.25m9-5.25v9l-9 5.25M3 7.5l9 5.25M3 7.5v9l9 5.25" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
            <p style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.18em", textTransform: "uppercase" as const, color: props.darkMode ? "rgba(180,165,220,0.8)" : "rgba(100,95,90,0.8)", margin: 0 }}>
              No Active Model
            </p>
            <p style={{ fontSize: 13, lineHeight: 1.65, color: props.darkMode ? "rgba(200,190,230,0.7)" : "rgba(80,75,70,0.7)", margin: "8px 0 0" }}>
              Upload an image or JSON file to render the interactive 3D model.
            </p>
          </div>
        </div>
      )}
      {/* Viewer label — bottom left */}
      <div style={{
        position: "absolute",
        left: 12,
        bottom: 12,
        borderRadius: 8,
        border: `1px solid ${props.darkMode ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.07)"}`,
        background: props.darkMode ? "rgba(14,12,22,0.75)" : "rgba(255,254,250,0.82)",
        backdropFilter: "blur(8px)",
        padding: "4px 10px",
        fontSize: 10,
        fontWeight: 700,
        letterSpacing: "0.16em",
        textTransform: "uppercase" as const,
        color: props.darkMode ? "rgba(180,170,220,0.8)" : "rgba(80,75,70,0.75)",
        pointerEvents: "none" as const,
      }}>
        3D Viewer
      </div>
      {/* Pick mode controls — top right */}
      <div style={{ position: "absolute", right: 12, top: 12, zIndex: 20, display: "flex", gap: 4 }}>
        {(["auto", "walls", "nodes"] as const).map((mode) => (
          <button
            key={mode}
            type="button"
            onClick={() => setPickMode(mode)}
            style={{
              borderRadius: 7,
              border: `1px solid ${pickMode === mode
                ? (props.darkMode ? "rgba(130,200,255,0.4)" : "rgba(50,130,200,0.35)")
                : (props.darkMode ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)")}`,
              background: pickMode === mode
                ? (props.darkMode ? "rgba(80,160,240,0.18)" : "rgba(50,130,200,0.12)")
                : (props.darkMode ? "rgba(14,12,22,0.7)" : "rgba(255,254,250,0.8)"),
              backdropFilter: "blur(8px)",
              padding: "4px 10px",
              fontSize: 10,
              fontWeight: 600,
              letterSpacing: "0.1em",
              textTransform: "capitalize" as const,
              color: pickMode === mode
                ? (props.darkMode ? "rgba(150,220,255,0.95)" : "rgba(30,100,175,0.9)")
                : (props.darkMode ? "rgba(180,170,220,0.75)" : "rgba(80,75,70,0.75)"),
              cursor: "pointer",
              transition: "background 0.15s, border-color 0.15s, color 0.15s",
            }}
          >
            {mode.charAt(0).toUpperCase() + mode.slice(1)}
          </button>
        ))}
      </div>
      {hoverPreview && (
        <div
          style={{
            position: "absolute",
            zIndex: 20,
            left: hoverPreview.x + 14,
            top: hoverPreview.y + 14,
            pointerEvents: "none" as const,
            borderRadius: 7,
            border: `1px solid ${props.darkMode ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}`,
            background: props.darkMode ? "rgba(18,14,28,0.92)" : "rgba(255,254,250,0.94)",
            backdropFilter: "blur(8px)",
            padding: "4px 10px",
            fontSize: 11,
            fontWeight: 600,
            letterSpacing: "0.06em",
            color: props.darkMode ? "rgba(210,200,240,0.9)" : "rgba(50,45,40,0.9)",
          }}
        >
          {hoverPreview.type} <span style={{ opacity: 0.6 }}>{hoverPreview.id}</span>
        </div>
      )}
    </div>
  );
}
