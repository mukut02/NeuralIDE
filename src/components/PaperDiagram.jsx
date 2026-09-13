import { useState, useEffect, useMemo, useRef } from "react";
import {
  calculateLayerParams,
  calculateLayerFlops,
  calculateTensorBytes,
  formatBytes,
  formatParams,
  formatFlops,
  generateSyntheticFeatureMaps,
} from "../utils/architectureAnalytics.js";

// Vibrant, publication-grade color scheme for layer categories
const LAYER_COLORS = {
  // Convolutions & Feature extraction -> Electric Cyan / Cerulean
  Conv2d: { main: "#38bdf8", top: "#7dd3fc", side: "#0284c7", border: "#bae6fd", text: "#e0f2fe" },
  PointwiseConv: { main: "#38bdf8", top: "#7dd3fc", side: "#0284c7", border: "#bae6fd", text: "#e0f2fe" },
  DepthwiseConv: { main: "#06b6d4", top: "#67e8f9", side: "#0891b2", border: "#a5f3fc", text: "#cffafe" },
  DilatedConv: { main: "#0284c7", top: "#38bdf8", side: "#0369a1", border: "#7dd3fc", text: "#e0f2fe" },
  TransposedConv: { main: "#0ea5e9", top: "#38bdf8", side: "#0369a1", border: "#7dd3fc", text: "#e0f2fe" },

  // Activations -> Warm Amber / Gold
  ReLU: { main: "#f59e0b", top: "#fcd34d", side: "#d97706", border: "#fef3c7", text: "#fffbeb" },
  LeakyReLU: { main: "#f59e0b", top: "#fcd34d", side: "#d97706", border: "#fef3c7", text: "#fffbeb" },
  GELU: { main: "#fbbf24", top: "#fde68a", side: "#d97706", border: "#fef3c7", text: "#fffbeb" },
  SiLU: { main: "#fbbf24", top: "#fde68a", side: "#d97706", border: "#fef3c7", text: "#fffbeb" },
  Tanh: { main: "#f97316", top: "#fdba74", side: "#ea580c", border: "#ffedd5", text: "#fff7ed" },
  Sigmoid: { main: "#f97316", top: "#fdba74", side: "#ea580c", border: "#ffedd5", text: "#fff7ed" },

  // Pooling -> Purple / Violet
  MaxPool2d: { main: "#a855f7", top: "#d8b4fe", side: "#9333ea", border: "#f3e8ff", text: "#faf5ff" },
  AveragePool: { main: "#9333ea", top: "#c084fc", side: "#7e22ce", border: "#e9d5ff", text: "#faf5ff" },
  AdaptiveAvgPool: { main: "#8b5cf6", top: "#a78bfa", side: "#6d28d9", border: "#ddd6fe", text: "#f5f3ff" },
  GlobalAvgPool: { main: "#7c3aed", top: "#a78bfa", side: "#5b21b6", border: "#ddd6fe", text: "#f5f3ff" },

  // Normalization -> Emerald / Mint
  BatchNorm: { main: "#10b981", top: "#6ee7b7", side: "#059669", border: "#a7f3d0", text: "#ecfdf5" },
  LayerNorm: { main: "#059669", top: "#34d399", side: "#047857", border: "#a7f3d0", text: "#ecfdf5" },
  InstanceNorm: { main: "#10b981", top: "#6ee7b7", side: "#059669", border: "#a7f3d0", text: "#ecfdf5" },
  GroupNorm: { main: "#047857", top: "#10b981", side: "#065f46", border: "#6ee7b7", text: "#ecfdf5" },

  // Dense / Fully Connected -> Indigo / Royal Blue
  Linear: { main: "#6366f1", top: "#a5b4fc", side: "#4338ca", border: "#c7d2fe", text: "#eef2ff" },
  Dense: { main: "#6366f1", top: "#a5b4fc", side: "#4338ca", border: "#c7d2fe", text: "#eef2ff" },
  ClassifierHead: { main: "#4f46e5", top: "#818cf8", side: "#3730a3", border: "#c7d2fe", text: "#eef2ff" },

  // Regularization -> Rose / Vermillion
  Dropout: { main: "#f43f5e", top: "#fda4af", side: "#e11d48", border: "#ffe4e6", text: "#fff1f2" },
  Dropout2D: { main: "#f43f5e", top: "#fda4af", side: "#e11d48", border: "#ffe4e6", text: "#fff1f2" },
  DropPath: { main: "#e11d48", top: "#fb7185", side: "#be123c", border: "#ffe4e6", text: "#fff1f2" },

  // Graph Connections -> Bright Cyan / Lime
  Add: { main: "#14b8a6", top: "#5eead4", side: "#0d9488", border: "#ccfbf1", text: "#f0fdfa" },
  Residual: { main: "#14b8a6", top: "#5eead4", side: "#0d9488", border: "#ccfbf1", text: "#f0fdfa" },
  Concatenate: { main: "#0d9488", top: "#2dd4bf", side: "#0f766e", border: "#99f6e4", text: "#f0fdfa" },

  // Attention -> Coral / Crimson
  SelfAttention: { main: "#ec4899", top: "#f472b6", side: "#db2777", border: "#fce7f3", text: "#fdf2f8" },
  MultiHeadAttention: { main: "#ec4899", top: "#f472b6", side: "#db2777", border: "#fce7f3", text: "#fdf2f8" },

  // Default
  Default: { main: "#64748b", top: "#94a3b8", side: "#475569", border: "#cbd5e1", text: "#f8fafc" },
};

export default function PaperDiagram({
  architecture,
  dataset,
  connections = [],
  training,
  onSelectLayer,
  selectedId,
  usePretrainedWeights = false,
  currentPreset = null,
  onTogglePretrainedWeights,
  showToast,
}) {
  const [viewMode, setViewMode] = useState("isometric"); // "isometric" | "orthographic"
  const [theme, setTheme] = useState("dark"); // "dark" | "light" (Academic Paper Mode)
  const [activeStep, setActiveStep] = useState(0); // 0 = input, 1..N = layers
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1200); // ms per step
  const [selectedClass, setSelectedClass] = useState(() => {
    if (dataset?.name === "CIFAR-10") return 3; // Cat
    if (dataset?.name === "MNIST") return 7; // Digit 7
    return 0; // T-shirt / top
  });
  const svgRef = useRef(null);

  // Sync selectedClass default if dataset changes
  useEffect(() => {
    const defaultIdx = dataset?.name === "CIFAR-10" ? 3 : dataset?.name === "MNIST" ? 7 : 0;
    setSelectedClass(defaultIdx);
  }, [dataset?.name]);

  const totalSteps = architecture.length;

  // Auto playback of forward pass dry run
  useEffect(() => {
    let timer;
    if (isPlaying) {
      timer = setInterval(() => {
        setActiveStep((curr) => {
          if (curr >= totalSteps) {
            setIsPlaying(false);
            return totalSteps;
          }
          return curr + 1;
        });
      }, playbackSpeed);
    }
    return () => clearInterval(timer);
  }, [isPlaying, totalSteps, playbackSpeed]);

  // Aggregate model metrics
  const modelStats = useMemo(() => {
    let totalParams = 0;
    let totalFlops = 0;
    let peakMemoryBytes = 0;

    architecture.forEach((layer) => {
      const p = calculateLayerParams(layer, layer.input, layer.output);
      const f = calculateLayerFlops(layer, layer.input, layer.output);
      const mem = calculateTensorBytes(layer.output, 1);
      totalParams += p;
      totalFlops += f;
      if (mem > peakMemoryBytes) peakMemoryBytes = mem;
    });

    return {
      totalParams,
      totalFlops,
      peakMemoryBytes,
      numLayers: architecture.length,
    };
  }, [architecture]);

  // Current active step details
  const currentStepData = useMemo(() => {
    const currentClassLabel = dataset.classesList?.[selectedClass] || `Class ${selectedClass}`;
    if (activeStep === 0) {
      const inputShape = { kind: "image", c: dataset.channels, h: dataset.size, w: dataset.size };
      return {
        isInput: true,
        title: "Input Tensor",
        subtitle: `${dataset.name} · Sample: ${currentClassLabel}`,
        shape: inputShape,
        params: 0,
        flops: 0,
        memBytes: calculateTensorBytes(inputShape, 1),
        type: "Input",
      };
    }
    const layer = architecture[activeStep - 1];
    if (!layer) return null;
    return {
      isInput: false,
      layer,
      title: `Layer ${activeStep}: ${layer.type}`,
      subtitle: layer.params?.filters
        ? `${layer.params.filters} filters · ${layer.params.kernel || 3}×${layer.params.kernel || 3} receptive field`
        : layer.params?.units
          ? `${layer.params.units} hidden units`
          : layer.params?.heads
            ? `${layer.params.heads} self-attention heads`
            : "Structural Operation",
      shape: layer.output,
      params: calculateLayerParams(layer, layer.input, layer.output),
      flops: calculateLayerFlops(layer, layer.input, layer.output),
      memBytes: calculateTensorBytes(layer.output, 1),
      type: layer.type,
      stage: layer.stage,
    };
  }, [activeStep, architecture, dataset, selectedClass]);

  // Synthetic activation heatmaps for the current step
  const syntheticActivation = useMemo(() => {
    if (!currentStepData) return null;
    return generateSyntheticFeatureMaps(
      activeStep,
      totalSteps,
      currentStepData.shape,
      dataset,
      selectedClass
    );
  }, [currentStepData, activeStep, totalSteps, dataset, selectedClass]);

  // Export diagram as publication vector SVG
  const handleExportSvg = () => {
    const svgEl = svgRef.current;
    if (!svgEl) return;
    try {
      const serializer = new XMLSerializer();
      const source = serializer.serializeToString(svgEl);
      const svgBlob = `<?xml version="1.0" standalone="no"?>\r\n${source}`;
      const blob = new Blob([svgBlob], { type: "image/svg+xml;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `neuralide_${dataset.name.toLowerCase()}_architecture_diagram.svg`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      showToast?.("Model Insights diagram exported as vector SVG!");
    } catch {
      showToast?.("Failed to export SVG diagram.");
    }
  };

  // Diagram geometry calculation
  const nodeWidth = 96;
  const nodeSpacing = 72;
  const svgWidth = Math.max(1200, (architecture.length + 2) * (nodeWidth + nodeSpacing));
  const svgHeight = 440;
  const centerY = 200;

  // Group layers into architectural stages with robust fallback
  const stageGroups = useMemo(() => {
    const groups = [];
    let currentGroup = null;

    architecture.forEach((layer, index) => {
      let stageName = layer.stage;
      if (!stageName) {
        if (layer.type === "Linear" || layer.type === "ClassifierHead" || layer.type === "Dropout" || layer.type === "Flatten") {
          stageName = "Classifier Head";
        } else if (layer.input?.kind === "image") {
          stageName = `Stage (${layer.input.h}×${layer.input.w || layer.input.h})`;
        } else {
          stageName = "Dense Representation";
        }
      }

      if (!currentGroup || currentGroup.name !== stageName) {
        currentGroup = {
          name: stageName,
          startIndex: index,
          endIndex: index,
        };
        groups.push(currentGroup);
      } else {
        currentGroup.endIndex = index;
      }
    });

    return groups;
  }, [architecture]);

  const isLight = theme === "light";

  return (
    <div className={`paper-diagram-container ${isLight ? "paper-light-theme" : "paper-dark-theme"}`}>
      {/* Top Banner: Publication Metrics & Diagram Controls */}
      <div className="paper-diagram-header">
        <div className="paper-title-group">
          <span className="paper-eyebrow">MODEL INSIGHTS · NEURAL NETWORK FLOW & ACTIVATION DYNAMICS</span>
          <h2>Model Insights & Live Tensor Dry Run</h2>
        </div>

        {/* Aggregate Stats Badges */}
        <div className="paper-stats-pills">
          <div className="stat-pill">
            <span className="stat-label">PARAM COUNT</span>
            <strong className="stat-value">{formatParams(modelStats.totalParams)}</strong>
          </div>
          <div className="stat-pill">
            <span className="stat-label">PEAK MEMORY</span>
            <strong className="stat-value">{formatBytes(modelStats.peakMemoryBytes)}</strong>
          </div>
          <div className="stat-pill">
            <span className="stat-label">COMPUTATIONAL FLOPs</span>
            <strong className="stat-value">{formatFlops(modelStats.totalFlops)}</strong>
          </div>
          <div className="stat-pill">
            <span className="stat-label">TRAINING OPTIMIZER</span>
            <strong className="stat-value">{training?.optimizer || "AdamW"}</strong>
          </div>
          <div
            className={`stat-pill ${onTogglePretrainedWeights ? "clickable-pill" : ""}`}
            onClick={onTogglePretrainedWeights}
            title={usePretrainedWeights ? "Pre-trained ImageNet weights active. Click to switch to scratch." : "Scratch initialization. Click to load pre-trained weights."}
            style={{ cursor: onTogglePretrainedWeights ? "pointer" : "default" }}
          >
            <span className="stat-label">WEIGHTS</span>
            <strong className="stat-value" style={{ color: usePretrainedWeights ? "#9de5c1" : "#79b7f4" }}>
              {usePretrainedWeights && currentPreset?.pretrainedWeights ? "Pre-trained" : "Scratch"}
            </strong>
          </div>
        </div>

        {/* View Mode & Export Buttons */}
        <div className="paper-actions">
          {/* Academic Theme Switcher */}
          <button
            type="button"
            className={`paper-theme-btn ${isLight ? "light-mode" : "dark-mode"}`}
            onClick={() => setTheme(isLight ? "dark" : "light")}
            title="Toggle between Academic White Paper Mode and Dark Studio Theme"
          >
            {isLight ? (
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
              </svg>
            ) : (
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
              </svg>
            )}
            <span>{isLight ? "Academic Paper" : "Dark Studio"}</span>
          </button>

          <div className="view-mode-toggle" role="radiogroup" aria-label="Diagram perspective">
            <button
              type="button"
              className={`toggle-mode-btn ${viewMode === "isometric" ? "active" : ""}`}
              onClick={() => setViewMode("isometric")}
              title="3D Isometric Block Perspective"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
                <polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/>
              </svg>
              <span>3D Isometric</span>
            </button>
            <button
              type="button"
              className={`toggle-mode-btn ${viewMode === "orthographic" ? "active" : ""}`}
              onClick={() => setViewMode("orthographic")}
              title="2D Orthographic Blueprint Mode"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/>
              </svg>
              <span>2D Blueprint</span>
            </button>
          </div>

          <button
            type="button"
            className="paper-export-btn"
            onClick={handleExportSvg}
            title="Download publication-quality SVG graphic"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
            <span>Export SVG</span>
          </button>
        </div>
      </div>

      {/* Dry Run Interactive Simulation Controller Bar */}
      <div className="dry-run-controller">
        <div className="dry-run-controls">
          <button
            type="button"
            className="ctrl-step-btn"
            onClick={() => setActiveStep(0)}
            disabled={activeStep === 0}
            title="Rewind to input tensor"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="11 19 2 12 11 5 11 19"/><polygon points="22 19 13 12 22 5 22 19"/>
            </svg>
          </button>
          <button
            type="button"
            className="ctrl-step-btn"
            onClick={() => setActiveStep((c) => Math.max(0, c - 1))}
            disabled={activeStep === 0}
            title="Step backward"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="19 20 9 12 19 4 19 20"/><line x1="5" y1="19" x2="5" y2="5"/>
            </svg>
          </button>

          <button
            type="button"
            className={`ctrl-play-btn ${isPlaying ? "playing" : ""}`}
            onClick={() => {
              if (activeStep >= totalSteps && !isPlaying) {
                setActiveStep(0);
              }
              setIsPlaying(!isPlaying);
            }}
            title={isPlaying ? "Pause forward simulation" : "Play forward pass dry run"}
          >
            {isPlaying ? (
              <>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="6" y1="4" x2="6" y2="20"/><line x1="18" y1="4" x2="18" y2="20"/>
                </svg>
                <span>Pause</span>
              </>
            ) : (
              <>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                  <polygon points="5 3 19 12 5 21 5 3"/>
                </svg>
                <span>Simulate Forward Pass</span>
              </>
            )}
          </button>

          <button
            type="button"
            className="ctrl-step-btn"
            onClick={() => setActiveStep((c) => Math.min(totalSteps, c + 1))}
            disabled={activeStep >= totalSteps}
            title="Step forward"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="5 4 15 12 5 20 5 4"/><line x1="19" y1="5" x2="19" y2="19"/>
            </svg>
          </button>
          <button
            type="button"
            className="ctrl-step-btn"
            onClick={() => setActiveStep(totalSteps)}
            disabled={activeStep >= totalSteps}
            title="Fast forward to prediction output"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="13 19 22 12 13 5 13 19"/><polygon points="2 19 11 12 2 5 2 19"/>
            </svg>
          </button>
        </div>

        {/* Step Scrubber */}
        <div className="dry-run-scrubber">
          <span className="step-count">
            Step <strong>{activeStep}</strong> of {totalSteps}
          </span>
          <input
            type="range"
            min="0"
            max={totalSteps}
            value={activeStep}
            onChange={(e) => setActiveStep(Number(e.target.value))}
            className="scrubber-slider"
            aria-label="Forward pass step scrubber"
          />
          <span className="step-name">
            {activeStep === 0 ? "Input Tensor" : architecture[activeStep - 1]?.type}
          </span>
        </div>

        {/* Sample Class Quick Switcher */}
        <div className="dry-run-class-select-group">
          <span className="class-select-label">Class:</span>
          <select
            className="dry-run-class-dropdown"
            value={selectedClass}
            onChange={(e) => setSelectedClass(Number(e.target.value))}
            title="Switch dataset sample class for dry run forward pass"
          >
            {(dataset.classesList || []).map((name, idx) => (
              <option key={idx} value={idx}>
                {idx}: {name}
              </option>
            ))}
          </select>
        </div>

        {/* Speed Selector */}
        <div className="dry-run-speed">
          <span className="speed-label">Speed:</span>
          {[
            { label: "0.5×", val: 1800 },
            { label: "1×", val: 1200 },
            { label: "2×", val: 600 },
          ].map((s) => (
            <button
              key={s.label}
              type="button"
              className={`speed-pill ${playbackSpeed === s.val ? "active" : ""}`}
              onClick={() => setPlaybackSpeed(s.val)}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      {/* Main Diagram Canvas: Publication-Grade Interactive SVG */}
      <div className="paper-canvas-viewport">
        <svg
          ref={svgRef}
          className="paper-diagram-svg"
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          xmlns="http://www.w3.org/2000/svg"
        >
          <defs>
            {/* Ambient Background Grid Pattern */}
            <pattern id="paperGrid" width="24" height="24" patternUnits="userSpaceOnUse">
              <path d="M 24 0 L 0 0 0 24" fill="none" stroke={isLight ? "rgba(203, 213, 225, 0.4)" : "rgba(42, 68, 98, 0.25)"} strokeWidth="0.8" />
            </pattern>

            {/* Glowing Drop Shadow Filter for Active Step */}
            <filter id="activeGlow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="6" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>

            {/* Receptive Field Projection Line Gradient */}
            <linearGradient id="frustumGrad" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="rgba(121, 183, 244, 0.45)" />
              <stop offset="100%" stopColor="rgba(157, 229, 193, 0.45)" />
            </linearGradient>

            {/* Arcing Skip Connection Marker */}
            <marker id="skipArrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 10 5 L 0 9 z" fill="#8dd9bd" />
            </marker>
          </defs>

          {/* Background Canvas */}
          <rect width={svgWidth} height={svgHeight} fill={isLight ? "#f8fafc" : "#071221"} />
          <rect width={svgWidth} height={svgHeight} fill="url(#paperGrid)" opacity={isLight ? 0.9 : 0.7} />

          {/* Render Hierarchical Stage Bounding Containers */}
          {stageGroups.map((group, gIdx) => {
            const startX = (group.startIndex + 1) * (nodeWidth + nodeSpacing) - 20;
            const endX = (group.endIndex + 1) * (nodeWidth + nodeSpacing) + nodeWidth + 20;
            const w = endX - startX;
            const boxY = 28;
            const boxH = svgHeight - 56;
            const maxTagW = Math.max(44, Math.min(w - 18, 240));
            const maxChars = Math.max(3, Math.floor((maxTagW - 16) / 6.4));
            const displayName = group.name.length > maxChars
              ? group.name.slice(0, Math.max(2, maxChars - 1)).trim() + "…"
              : group.name;

            return (
              <g key={`stage-${gIdx}`} className="diagram-stage-group">
                <defs>
                  <clipPath id={`stage-clip-${gIdx}`}>
                    <rect x={startX + 10} y={boxY + 6} width={maxTagW} height="20" rx="5" />
                  </clipPath>
                </defs>
                <rect
                  x={startX}
                  y={boxY}
                  width={w}
                  height={boxH}
                  rx="10"
                  fill={isLight ? "rgba(241, 245, 249, 0.85)" : "rgba(15, 23, 42, 0.45)"}
                  stroke={isLight ? "rgba(203, 213, 225, 0.9)" : "rgba(56, 189, 248, 0.18)"}
                  strokeWidth="1.2"
                />
                {/* Stage Header Tag */}
                <rect
                  x={startX + 10}
                  y={boxY + 6}
                  width={maxTagW}
                  height="20"
                  rx="5"
                  fill={isLight ? "#e2e8f0" : "rgba(30, 41, 59, 0.9)"}
                  stroke={isLight ? "#cbd5e1" : "rgba(56, 189, 248, 0.3)"}
                  strokeWidth="0.8"
                />
                <text
                  x={startX + 18}
                  y={boxY + 20}
                  clipPath={`url(#stage-clip-${gIdx})`}
                  fill={isLight ? "#1e293b" : "#7dd3fc"}
                  fontSize="9.5"
                  fontWeight="700"
                  letterSpacing="0.04em"
                  fontFamily="ui-monospace, monospace"
                >
                  {displayName.toUpperCase()}
                  <title>{group.name}</title>
                </text>
              </g>
            );
          })}

          {/* Render Connections / Residual Skip Arcs */}
          {connections.map((conn) => {
            const fromIndex = architecture.findIndex((item) => item.id === conn.from);
            const toIndex = architecture.findIndex((item) => item.id === conn.to);
            if (fromIndex < 0 || toIndex < 0) return null;

            const x1 = (fromIndex + 1) * (nodeWidth + nodeSpacing) + 50;
            const x2 = (toIndex + 1) * (nodeWidth + nodeSpacing) + 50;
            const arcHeight = 75 + Math.abs(toIndex - fromIndex) * 12;
            const yArc = centerY - arcHeight;

            return (
              <g key={`conn-${conn.from}-${conn.to}`} className="skip-connection-arc">
                <path
                  d={`M ${x1} ${centerY - 45} C ${x1} ${yArc}, ${x2} ${yArc}, ${x2} ${centerY - 45}`}
                  fill="none"
                  stroke="#8dd9bd"
                  strokeWidth="2"
                  strokeDasharray="4,4"
                  markerEnd="url(#skipArrow)"
                />
                <rect x={(x1 + x2) / 2 - 32} y={yArc - 12} width="64" height="18" rx="4" fill={isLight ? "#ffffff" : "#0c2336"} stroke={isLight ? "#cbd5e1" : "#25556f"} strokeWidth="1" />
                <text x={(x1 + x2) / 2} y={yArc + 1} textAnchor="middle" fill="#10b981" fontSize="9.5" fontWeight="700" fontFamily="ui-monospace, monospace">
                  Identity Skip
                </text>
              </g>
            );
          })}

          {/* Node 0: Input Tensor Block */}
          {(() => {
            const x = 50;
            const w = dataset.size === 32 ? 46 : 40;
            const h = dataset.size === 32 ? 46 : 40;
            const d = dataset.channels === 3 ? 20 : 12;
            const isActive = activeStep === 0;

            return (
              <g
                key="input-node"
                className={`diagram-node ${isActive ? "active-step" : ""}`}
                onClick={() => setActiveStep(0)}
                style={{ cursor: "pointer" }}
              >
                {/* Receptive Frustum line to Layer 1 */}
                <path
                  d={`M ${x + w + d * 0.7} ${centerY - h / 2} L ${x + nodeWidth + nodeSpacing - 10} ${centerY - 25} L ${x + nodeWidth + nodeSpacing - 10} ${centerY + 25} L ${x + w} ${centerY + h / 2} Z`}
                  fill="url(#frustumGrad)"
                  opacity={isLight ? "0.08" : "0.15"}
                />

                {/* Block 3D Isometric or 2D */}
                {viewMode === "isometric" ? (
                  <IsometricCuboid
                    x={x}
                    y={centerY - h / 2}
                    w={w}
                    h={h}
                    d={d}
                    colors={{ main: "#4f82af", top: "#79aada", side: "#2b567d", border: "#9ec5ec" }}
                    isActive={isActive}
                    isLight={isLight}
                  />
                ) : (
                  <rect
                    x={x}
                    y={centerY - h / 2}
                    width={w}
                    height={h}
                    rx="4"
                    fill="#4f82af"
                    stroke={isActive ? "#ffffff" : "#9ec5ec"}
                    strokeWidth={isActive ? 2.5 : 1}
                  />
                )}

                {/* Text Labels */}
                <text x={x + w / 2} y={centerY - 45} textAnchor="middle" fill={isLight ? "#0f172a" : "#eaf2fc"} fontSize="11" fontWeight="700" fontFamily="ui-sans-serif, system-ui">
                  INPUT
                </text>
                <text x={x + w / 2} y={centerY - 32} textAnchor="middle" fill={isLight ? "#475569" : "#7593b4"} fontSize="9.5" fontFamily="ui-monospace, monospace">
                  {dataset.name}
                </text>

                {/* Dimension Badge */}
                <rect x={x + w / 2 - 38} y={centerY + 45} width="76" height="20" rx="4" fill={isLight ? "#ffffff" : "#0a1a2c"} stroke={isActive ? "#38bdf8" : isLight ? "#cbd5e1" : "#1f3752"} strokeWidth="1" />
                <text x={x + w / 2} y={centerY + 59} textAnchor="middle" fill={isActive ? "#0284c7" : isLight ? "#334155" : "#79b7f4"} fontSize="9.5" fontWeight="600" fontFamily="ui-monospace, monospace">
                  {dataset.channels}×{dataset.size}×{dataset.size}
                </text>
              </g>
            );
          })()}

          {/* Sequential Layers */}
          {architecture.map((layer, index) => {
            const stepNum = index + 1;
            const isActive = activeStep === stepNum;
            const isSelected = selectedId === layer.id;
            const colors = LAYER_COLORS[layer.type] || LAYER_COLORS.Default;

            const x = (index + 1) * (nodeWidth + nodeSpacing) + 50;
            const prevX = index * (nodeWidth + nodeSpacing) + 50;

            // Geometry sizing based on output shape
            const isImage = layer.output?.kind === "image";
            const spatialH = isImage ? Math.max(22, Math.min(64, layer.output.h * 1.5)) : 32;
            const spatialW = isImage ? Math.max(22, Math.min(64, layer.output.w * 1.5)) : 32;
            const channelD = isImage ? Math.max(12, Math.min(38, Math.log2(layer.output.c || 32) * 5.5)) : 14;

            const shapeStr = isImage
              ? `${layer.output.c}×${layer.output.h}×${layer.output.w}`
              : `${layer.output.n}`;

            return (
              <g
                key={`layer-node-${layer.id}`}
                className={`diagram-node ${isActive ? "active-step" : ""} ${isSelected ? "selected-layer" : ""}`}
                onClick={() => {
                  setActiveStep(stepNum);
                  onSelectLayer?.(layer.id);
                }}
                style={{ cursor: "pointer" }}
              >
                {/* Connecting Inter-Layer Tensor Bridge */}
                <line
                  x1={prevX + 50}
                  y1={centerY}
                  x2={x}
                  y2={centerY}
                  stroke={isActive ? colors.main : isLight ? "#cbd5e1" : "#223d5a"}
                  strokeWidth={isActive ? 2.5 : 1.5}
                  strokeDasharray={layer.type === "Flatten" ? "3,3" : "none"}
                />

                {/* Projection Frustum if Conv/Pool */}
                {["Conv2d", "MaxPool2d", "AveragePool"].includes(layer.type) && (
                  <path
                    d={`M ${prevX + 46} ${centerY - 16} L ${x} ${centerY - spatialH / 2} L ${x} ${centerY + spatialH / 2} L ${prevX + 46} ${centerY + 16} Z`}
                    fill={colors.main}
                    opacity={isLight ? "0.05" : "0.08"}
                  />
                )}

                {/* 3D Isometric Cuboid or 2D Box */}
                {isImage ? (
                  viewMode === "isometric" ? (
                    <IsometricCuboid
                      x={x}
                      y={centerY - spatialH / 2}
                      w={spatialW}
                      h={spatialH}
                      d={channelD}
                      colors={colors}
                      isActive={isActive}
                      isLight={isLight}
                    />
                  ) : (
                    <rect
                      x={x}
                      y={centerY - spatialH / 2}
                      width={spatialW}
                      height={spatialH}
                      rx="4"
                      fill={colors.main}
                      stroke={isActive ? "#ffffff" : colors.border}
                      strokeWidth={isActive ? 2.5 : 1}
                      fillOpacity="0.85"
                    />
                  )
                ) : (
                  /* Dense 1D Feature Vector Bar Representation */
                  <g className="dense-vector-block">
                    <rect
                      x={x}
                      y={centerY - 32}
                      width={24}
                      height={64}
                      rx="5"
                      fill={colors.side}
                      stroke={isActive ? "#ffffff" : colors.border}
                      strokeWidth={isActive ? 2.5 : 1.2}
                    />
                    {/* Feature tick lines */}
                    {[centerY - 20, centerY - 8, centerY + 4, centerY + 16].map((tickY, i) => (
                      <line key={i} x1={x + 5} y1={tickY} x2={x + 19} y2={tickY} stroke={colors.top} strokeWidth="1.5" />
                    ))}
                  </g>
                )}

                {/* Layer Name & Type Label Above */}
                <text x={x + spatialW / 2} y={centerY - 45} textAnchor="middle" fill={isLight ? "#0f172a" : "#eaf2fc"} fontSize="11" fontWeight="700" fontFamily="ui-sans-serif, system-ui">
                  {layer.type}
                </text>
                <text x={x + spatialW / 2} y={centerY - 32} textAnchor="middle" fill={isLight ? "#475569" : "#7593b4"} fontSize="9.5" fontFamily="ui-monospace, monospace">
                  L{stepNum} · {layer.params?.filters ? `${layer.params.filters}f` : layer.params?.units ? `${layer.params.units}u` : ""}
                </text>

                {/* Mathematical Output Dimension Badge Below */}
                <rect x={x + spatialW / 2 - 42} y={centerY + 45} width="84" height="20" rx="4" fill={isLight ? "#ffffff" : "#0a1a2c"} stroke={isActive ? colors.main : isLight ? "#cbd5e1" : "#1f3752"} strokeWidth="1" />
                <text x={x + spatialW / 2} y={centerY + 59} textAnchor="middle" fill={isActive ? colors.main : isLight ? "#334155" : "#94b9df"} fontSize="9" fontWeight="600" fontFamily="ui-monospace, monospace">
                  {shapeStr}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Interactive Dry Run Tensor Inspector & Feature Heatmap Panel */}
      {currentStepData && (
        <div className="dry-run-inspector-card">
          {/* Step Meta Column */}
          <div className="inspector-step-col">
            <div className="step-badge-row">
              <span className="step-num-pill">
                {activeStep === 0 ? "INPUT" : `STEP ${activeStep} OF ${totalSteps}`}
              </span>
              <span className="step-type-pill" style={{ color: LAYER_COLORS[currentStepData.type]?.top || "#79b7f4" }}>
                {currentStepData.type}
              </span>
            </div>
            <h3 className="step-title">{currentStepData.title}</h3>
            <p className="step-subtitle">{currentStepData.subtitle}</p>

            {/* Interactive Dataset Class Selection Box in Input Section */}
            {activeStep === 0 ? (
              <div className="dry-run-class-box">
                <div className="class-box-header">
                  <span className="class-box-title">INPUT SAMPLE CLASS (DRY RUN)</span>
                  <span className="class-badge-active">
                    {dataset.classesList?.[selectedClass] || `Class ${selectedClass}`}
                  </span>
                </div>
                <p className="class-box-hint">
                  Select a class to inject authentic sample patterns and inspect real-time feature activations:
                </p>
                <div className="class-pills-wrap">
                  {(dataset.classesList || []).map((name, idx) => {
                    const isSelected = selectedClass === idx;
                    return (
                      <button
                        key={idx}
                        type="button"
                        className={`dry-run-class-chip ${isSelected ? "active" : ""}`}
                        onClick={() => setSelectedClass(idx)}
                        title={`Select ${name} (Class ${idx})`}
                      >
                        <span className="chip-index">{idx}</span>
                        <span className="chip-name">{name}</span>
                      </button>
                    );
                  })}
                </div>
                <button
                  type="button"
                  className="dry-run-simulate-btn"
                  onClick={() => {
                    setActiveStep(0);
                    setIsPlaying(true);
                  }}
                  title="Simulate forward pass through all layers with this class sample"
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                    <polygon points="5 3 19 12 5 21 5 3"/>
                  </svg>
                  <span>Simulate Forward Pass</span>
                </button>
              </div>
            ) : (
              <div className="step-class-reminder">
                <div className="reminder-left">
                  <span className="reminder-label">Simulating Class:</span>
                  <strong className="reminder-val">{dataset.classesList?.[selectedClass] || `Class ${selectedClass}`}</strong>
                </div>
                <button
                  type="button"
                  className="reminder-change-btn"
                  onClick={() => setActiveStep(0)}
                  title="Return to Input Tensor to switch sample class"
                >
                  Change Class
                </button>
              </div>
            )}

            <div className="step-specs-grid">
              <div className="spec-item">
                <span className="spec-label">OUTPUT SHAPE</span>
                <strong className="spec-val">
                  {currentStepData.shape?.kind === "image"
                    ? `(1, ${currentStepData.shape.c}, ${currentStepData.shape.h}, ${currentStepData.shape.w})`
                    : `(1, ${currentStepData.shape?.n || "?"})`}
                </strong>
              </div>
              <div className="spec-item">
                <span className="spec-label">TENSOR MEMORY</span>
                <strong className="spec-val">{formatBytes(currentStepData.memBytes)}</strong>
              </div>
              <div className="spec-item">
                <span className="spec-label">PARAMETERS</span>
                <strong className="spec-val">{formatParams(currentStepData.params)}</strong>
              </div>
              <div className="spec-item">
                <span className="spec-label">THEORETICAL FLOPs</span>
                <strong className="spec-val">{formatFlops(currentStepData.flops)}</strong>
              </div>
            </div>
          </div>

          {/* Activation Heatmap / Feature Visualizer Column */}
          <div className="inspector-heatmap-col">
            <div className="heatmap-header">
              <div className="heatmap-title-group">
                <span className="heatmap-title">
                  {syntheticActivation?.kind === "image"
                    ? `Simulated Feature Activations (${syntheticActivation.previewChannels} of ${syntheticActivation.totalChannels} channels)`
                    : syntheticActivation?.targetClass !== null
                      ? "Prediction Softmax Probabilities (Output Logits)"
                      : `Dense Feature Vector (${currentStepData.shape?.n || "?"} dimensions)`}
                </span>
                <span className="heatmap-class-badge">
                  Active Sample: <strong>{syntheticActivation?.classLabel || dataset.classesList?.[selectedClass] || `Class ${selectedClass}`}</strong>
                </span>
              </div>
              <span className="heatmap-scale">
                {syntheticActivation?.kind === "image"
                  ? "Intensity 0.0 → 1.0"
                  : syntheticActivation?.targetClass !== null
                    ? "Softmax Probability (0% → 100%)"
                    : "Activation Amplitude (0.0 → 1.0)"}
              </span>
            </div>

            {syntheticActivation?.kind === "image" ? (
              <div className="feature-slices-row">
                {syntheticActivation.slices.map((slice, sliceIdx) => {
                  const isCifarInput = dataset.name === "CIFAR-10" && activeStep === 0;
                  const channelTheme = isCifarInput
                    ? sliceIdx === 0
                      ? { label: "ch 1 (Red Channel)", color: "rgba(239, 68, 68,", border: "#ef4444", text: "#fca5a5" }
                      : sliceIdx === 1
                        ? { label: "ch 2 (Green Channel)", color: "rgba(34, 197, 94,", border: "#22c55e", text: "#86efac" }
                        : { label: "ch 3 (Blue Channel)", color: "rgba(56, 189, 248,", border: "#38bdf8", text: "#7dd3fc" }
                    : {
                        label: syntheticActivation.channelNames?.[sliceIdx] || `ch ${sliceIdx + 1}`,
                        color: "rgba(56, 189, 248,",
                        border: "#38bdf8",
                        text: "#93c5fd",
                      };

                  return (
                    <div key={sliceIdx} className="feature-slice-card" style={{ borderColor: isCifarInput ? channelTheme.border + "55" : undefined }}>
                      <span className="slice-label" style={{ color: channelTheme.text }}>
                        {channelTheme.label}
                      </span>
                      <div
                        className="slice-grid"
                        style={{
                          gridTemplateColumns: `repeat(${slice[0].length}, 1fr)`,
                        }}
                      >
                        {slice.map((row, rIdx) =>
                          row.map((val, cIdx) => (
                            <div
                              key={`${rIdx}-${cIdx}`}
                              className="heatmap-cell"
                              style={{
                                backgroundColor: `${channelTheme.color} ${Math.max(0.04, val)})`,
                                boxShadow: val > 0.65 ? `0 0 5px ${channelTheme.border}88` : "none",
                              }}
                              title={`${channelTheme.label}[${rIdx},${cIdx}] = ${val.toFixed(2)}`}
                            />
                          ))
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : syntheticActivation?.kind === "vector" ? (
              <div className="vector-distribution-view">
                <div className="logits-bars-container">
                  {syntheticActivation.values.map((val, idx) => {
                    const isTarget = syntheticActivation.targetClass === idx;
                    const isSoftmax = syntheticActivation.targetClass !== null;
                    const rawLabel = isSoftmax
                      ? (dataset?.classesList?.[idx] || `Class ${idx}`)
                      : `Feature dim ${idx + 1}`;
                    const shortName = isSoftmax
                      ? (rawLabel.includes("-") ? rawLabel.split("-")[1].trim().slice(0, 5) : rawLabel.slice(0, 5))
                      : `d${idx + 1}`;
                    const displayPercent = isSoftmax
                      ? `${(val * 100).toFixed(0)}%`
                      : val.toFixed(2);

                    return (
                      <div key={idx} className="logit-bar-wrapper" title={`${rawLabel}: ${isSoftmax ? (val * 100).toFixed(1) + "%" : val.toFixed(3)}`}>
                        <div className="logit-bar-track">
                          <div
                            className={`logit-bar-fill ${isTarget ? "target-class" : ""}`}
                            style={{ height: `${Math.round(val * 100)}%` }}
                          />
                        </div>
                        <span className={`logit-label ${isTarget ? "target-text" : ""}`}>{shortName}</span>
                        <span className="logit-prob">{displayPercent}</span>
                      </div>
                    );
                  })}
                </div>
                {syntheticActivation.targetClass !== null && (
                  <div className="target-class-banner">
                    <div className="banner-left">
                      <span className="target-check-icon">✓</span>
                      <span>Target Ground Truth: <strong>{syntheticActivation.targetClassLabel}</strong></span>
                    </div>
                    <div className="banner-right">
                      <span>Top-1 Probability: <strong>{((syntheticActivation.values[syntheticActivation.targetClass] || 0) * 100).toFixed(1)}%</strong></span>
                      <span className="target-match-pill">PREDICTED MATCH</span>
                    </div>
                  </div>
                )}
              </div>
            ) : null}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Renders a publication-grade 3D Isometric Cuboid in SVG with depth slice striations & drop shadows
 */
function IsometricCuboid({ x, y, w, h, d, colors, isActive, isLight = false }) {
  // Extrusion offsets for isometric 3D perspective
  const dx = d * 0.75;
  const dy = -d * 0.45;

  // Drop Shadow Ellipse under the block
  const shadowCx = x + w / 2 + dx / 2;
  const shadowCy = y + h + 12;
  const shadowRx = Math.max(14, w / 2 + dx / 2 + 4);

  // Front face
  const frontPath = `M ${x} ${y} L ${x + w} ${y} L ${x + w} ${y + h} L ${x} ${y + h} Z`;
  // Top face
  const topPath = `M ${x} ${y} L ${x + dx} ${y + dy} L ${x + w + dx} ${y + dy} L ${x + w} ${y} Z`;
  // Right side face
  const sidePath = `M ${x + w} ${y} L ${x + w + dx} ${y + dy} L ${x + w + dx} ${y + h + dy} L ${x + w} ${y + h} Z`;

  // Multi-plane channel depth slices for enhanced 3D texture
  const numSlices = d >= 24 ? 2 : d >= 16 ? 1 : 0;
  const sliceElements = [];
  for (let s = 1; s <= numSlices; s++) {
    const frac = s / (numSlices + 1);
    const sx = dx * frac;
    const sy = dy * frac;
    // Top slice line
    const topSlice = `M ${x + sx} ${y + sy} L ${x + w + sx} ${y + sy}`;
    // Side slice line
    const sideSlice = `M ${x + w + sx} ${y + sy} L ${x + w + sx} ${y + h + sy}`;
    sliceElements.push(
      <path
        key={`slice-${s}`}
        d={`${topSlice} ${sideSlice}`}
        fill="none"
        stroke={colors.border}
        strokeWidth="0.75"
        strokeOpacity="0.6"
        strokeDasharray="2,2"
      />
    );
  }

  return (
    <g className="isometric-cuboid" filter={isActive ? "url(#activeGlow)" : undefined}>
      {/* Soft Drop Shadow Ellipse */}
      <ellipse
        cx={shadowCx}
        cy={shadowCy}
        rx={shadowRx}
        ry={5}
        fill={isLight ? "rgba(0, 0, 0, 0.08)" : "rgba(0, 0, 0, 0.45)"}
        opacity={isActive ? 0.9 : 0.6}
      />

      {/* Top Face (Lighter shade) */}
      <path d={topPath} fill={colors.top} stroke={colors.border} strokeWidth="1" fillOpacity={isActive ? 0.95 : 0.85} />

      {/* Side Face (Darker shade) */}
      <path d={sidePath} fill={colors.side} stroke={colors.border} strokeWidth="1" fillOpacity={isActive ? 0.95 : 0.85} />

      {/* Channel Depth Slice Striations */}
      {sliceElements}

      {/* Front Face (Primary color) */}
      <path
        d={frontPath}
        fill={colors.main}
        stroke={isActive ? "#ffffff" : colors.border}
        strokeWidth={isActive ? 2 : 1}
        fillOpacity={isActive ? 1 : 0.9}
      />
    </g>
  );
}
