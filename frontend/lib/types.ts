export type Point2D = [number, number];
export type Point3D = [number, number, number];

export type WallInput = {
  id?: string;
  start?: unknown;
  end?: unknown;
  position?: unknown;
  thickness?: unknown;
  height?: unknown;
  type?: unknown;
};

export type RoomInput = {
  id?: string;
  name?: unknown;
  polygon?: unknown;
};

export type LabelInput = {
  id?: string;
  text?: unknown;
  position?: unknown;
};

export type DoorInput = {
  id?: string;
  wallId?: unknown;
  offset?: unknown;
  position?: unknown;
  width?: unknown;
  height?: unknown;
  swing?: unknown;
};

export type WindowInput = {
  id?: string;
  wallId?: unknown;
  offset?: unknown;
  position?: unknown;
  width?: unknown;
  height?: unknown;
  sillHeight?: unknown;
  windowType?: unknown;
  type?: unknown;
};

export type SlabInput = {
  id?: string;
  name?: unknown;
  polygon?: unknown;
  coordinates?: unknown;
  centroid?: unknown;
};

export type OpeningInput = {
  id?: string;
  kind?: unknown;
  wallId?: unknown;
  offset?: unknown;
  position?: unknown;
  width?: unknown;
  height?: unknown;
  sillHeight?: unknown;
  swing?: unknown;
  windowType?: unknown;
  type?: unknown;
};

export type GraphNodeInput = {
  id?: string;
  position?: unknown;
  degree?: unknown;
  type?: unknown;
  connectedWallIds?: unknown;
  loadBearingWallIds?: unknown;
  likelyColumn?: unknown;
  color?: unknown;
};

export type ColumnInput = {
  id?: string;
  nodeId?: unknown;
  position?: unknown;
  width?: unknown;
  depth?: unknown;
  height?: unknown;
  color?: unknown;
  degree?: unknown;
  connectedWallIds?: unknown;
};

export type MetaInput = {
  unit?: unknown;
  wallHeight?: unknown;
  defaultWallThickness?: unknown;
};

export type RawSceneInput = {
  meta?: unknown;
  walls?: unknown;
  rooms?: unknown;
  slabs?: unknown;
  labels?: unknown;
  doors?: unknown;
  windows?: unknown;
  openings?: unknown;
  graphNodes?: unknown;
  columns?: unknown;
};

export type ValidationSeverity = "error" | "warning" | "info";
export type ConfidenceLevel = "high" | "medium" | "low";
export type DatasetReadiness = "valid" | "partial" | "invalid";

export type ValidationIssue = {
  id: string;
  severity: ValidationSeverity;
  message: string;
  path?: string;
  wallId?: string;
  roomId?: string;
  openingId?: string;
  fixType?: "opening_clash";
};

export type WallType = "outer" | "partition" | "semi_structural";
export type WallClassification = "load_bearing" | "partition" | "semi_structural";
export type OpeningKind = "door" | "window";
export type WindowPanelType = "single" | "double";
export type DoorSwing = "left" | "right";
export type SceneSelectionType = "wall" | "node" | "column" | "slab" | "door" | "window";
export type SceneSelection = {
  type: SceneSelectionType;
  id: string;
};

export type NormalizedWall = {
  id: string;
  start: Point2D;
  end: Point2D;
  start3D: Point3D;
  end3D: Point3D;
  thickness: number;
  height: number;
  type: WallType;
  inferredType: boolean;
  length: number;
  midpoint: Point3D;
  angle: number;
  roomId?: string;
  confidence: ConfidenceLevel;
};

export type NormalizedRoom = {
  id: string;
  name: string;
  polygon2D: Point2D[];
  polygon3D: Point3D[];
  span: number;
  spanLine: [Point3D, Point3D];
  centroid: Point3D;
  area: number;
  inferredSpan: boolean;
  confidence: ConfidenceLevel;
};

export type NormalizedGraphNode = {
  id: string;
  position: Point3D;
  degree: number;
  type: "terminal" | "corner" | "junction" | "inline";
  connectedWallIds: string[];
  loadBearingWallIds: string[];
  likelyColumn: boolean;
  color: string;
};

export type NormalizedColumn = {
  id: string;
  nodeId: string;
  position: Point3D;
  width: number;
  depth: number;
  height: number;
  color: string;
  degree: number;
  connectedWallIds: string[];
};

export type NormalizedLabel = {
  id: string;
  text: string;
  position: Point3D;
};

export type NormalizedOpening = {
  id: string;
  wallId: string;
  kind: OpeningKind;
  offset: number;
  width: number;
  height: number;
  position: Point3D;
  angle: number;
  sillHeight?: number;
  swing?: DoorSwing;
  panelType?: WindowPanelType;
};

export type SceneMeta = {
  unit: "meter";
  wallHeight: number;
  defaultWallThickness: number;
};

export type SceneBounds = {
  minX: number;
  maxX: number;
  minZ: number;
  maxZ: number;
  width: number;
  depth: number;
  center: Point3D;
};

export type SceneData = {
  meta: SceneMeta;
  walls: NormalizedWall[];
  rooms: NormalizedRoom[];
  labels: NormalizedLabel[];
  openings: NormalizedOpening[];
  graphNodes: NormalizedGraphNode[];
  columns: NormalizedColumn[];
  bounds: SceneBounds;
  readiness: DatasetReadiness;
};

export type ValidationResult = {
  data: SceneData;
  issues: ValidationIssue[];
  usedFallbackDataset: boolean;
};

export type MaterialRecord = {
  name: string;
  cost: number;
  strength: number;
  durability: number;
  ease: number;
  use: string;
};

export type Recommendation = {
  name: string;
  score: number;
  rationale: string;
  confidence: ConfidenceLevel;
};

export type StructuralElementType = "load_bearing_wall" | "partition_wall" | "slab" | "column";

export type MaterialOption = {
  material: string;
  rank: number;
  tradeoffScore: number;
  cost: number;
  strength: number;
  durability: number;
  ease: number;
  rationale: string;
};

export type ElementRecommendation = {
  elementType: StructuralElementType;
  weightJustification: string;
  formula: string;
  structuralConcerns: string[];
  options: MaterialOption[];
};

export type RecommendationFocus = {
  selectedType: SceneSelectionType | "none";
  selectedId: string;
  focusedElementType: StructuralElementType | "none";
  sizeSummary: string;
};

export type MaterialRecommendationTable = {
  source: "gemini" | "deterministic";
  model: string;
  generatedAt: string;
  recommendationCount: number;
  focus: RecommendationFocus;
  rows: ElementRecommendation[];
};

export type StructuralRole = "primary_support" | "secondary_support" | "partition";

export type HeuristicSeverity = "low" | "medium" | "high";
export type HeuristicConfidence = "low" | "medium" | "high";

export type HeuristicSuggestionType =
  | "SPAN_FIX"
  | "ALIGNMENT_FIX"
  | "LOAD_PATH_GAP"
  | "COLUMN_GAP"
  | "MATERIAL_OPT"
  | "WALL_REMOVAL";

export type HeuristicMetrics = Record<string, number | string | boolean>;

export type WallStructuralProfile = {
  wallId: string;
  role: StructuralRole;
  isLoadBearing: boolean;
  partOfAxisChain: boolean;
  partOfMajorAxisChain: boolean;
  evidence: string[];
};

export type HeuristicSuggestion = {
  id: string;
  type: HeuristicSuggestionType;
  severity: HeuristicSeverity;
  confidence: HeuristicConfidence;
  location: string;
  issue: string;
  suggestion: string;
  impact: string;
  impactBasis: string;
  evidence: string[];
  assumptions: string[];
  relatedWallIds: string[];
  relatedRoomId?: string;
  metrics?: HeuristicMetrics;
};

export type HeuristicReport = {
  generatedAt: string;
  assumptions: string[];
  defaults: {
    thickSupportM: number;
    supportSpanLimitM: number;
    masonryFallbackSpanLimitM: number;
    openingRatioLimit: number;
  };
  wallProfiles: WallStructuralProfile[];
  suggestions: HeuristicSuggestion[];
};

export type RecommendationApiResponse = {
  materialTable: MaterialRecommendationTable;
  heuristics: HeuristicReport;
};

export type SpanPreviewLine = {
  id: string;
  roomId?: string;
  from: Point3D;
  to: Point3D;
  confidence: HeuristicConfidence;
};

export type OptimizationActionKind =
  | "alignment_snap"
  | "opening_clash_fix"
  | "wall_removal"
  | "advisory";

export type OptimizationActionState = "idle" | "pending" | "blocked" | "resolved";

export type OptimizationPreviewLine = {
  id: string;
  from: Point3D;
  to: Point3D;
  color: string;
  label?: string;
};

export type OptimizationAction = {
  id: string;
  kind: OptimizationActionKind;
  state: OptimizationActionState;
  safeToApply: boolean;
  conflictGroupId?: string;
  relatedWallIds: string[];
  relatedRoomId?: string;
  relatedOpeningId?: string;
  sourceSuggestionId?: string;
  sourceIssueId?: string;
  confidence: HeuristicConfidence;
  severity: HeuristicSeverity;
  title: string;
  issue: string;
  suggestion: string;
  impact: string;
  impactBasis: string;
  evidence: string[];
  assumptions: string[];
};

export type RoomContext = {
  room?: NormalizedRoom;
  span: number;
  inferredSpan: boolean;
  confidence: ConfidenceLevel;
  spanLine?: [Point3D, Point3D];
};
