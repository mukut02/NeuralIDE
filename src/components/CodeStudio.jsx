import { useEffect, useMemo, useRef, useState } from "react";
import Prism from "prismjs";
import "prismjs/components/prism-python";
import {
  calculateLayerParams,
  calculateLayerFlops,
  calculateTensorBytes,
  formatBytes,
  formatParams,
  formatFlops,
} from "../utils/architectureAnalytics.js";

export default function CodeStudio({
  code,
  editableCode,
  codeEdited,
  autoSync,
  onToggleAutoSync,
  onSyncToCanvas,
  onResetCode,
  onCodeChange,
  onBackToArchitecture,
  dataset,
  layers,
  architecture,
  training,
  usePretrainedWeights = false,
  currentPreset = null,
  showToast,
}) {
  const [isEditing, setIsEditing] = useState(false);
  const [copied, setCopied] = useState(false);
  const textareaRef = useRef(null);
  const preRef = useRef(null);
  const gutterRef = useRef(null);

  const displayCode = codeEdited ? editableCode : code;

  // Split code into lines for line-numbering gutter
  const lines = useMemo(() => displayCode.split("\n"), [displayCode]);

  // Syntax highlight with Prism, ensuring empty trailing line is preserved and fallback is escaped
  const highlightedHtml = useMemo(() => {
    try {
      const codeToHighlight = displayCode.endsWith("\n") ? displayCode + " " : displayCode;
      const lang = Prism.languages.python || Prism.languages.clike || Prism.languages.javascript;
      return Prism.highlight(codeToHighlight, lang, "python");
    } catch {
      return displayCode
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
    }
  }, [displayCode]);

  // Aggregate model metrics matching Paper Diagram
  const modelStats = useMemo(() => {
    let totalParams = 0;
    let totalFlops = 0;
    let peakMemoryBytes = 0;

    (architecture || []).forEach((layer) => {
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
      numLayers: architecture?.length || layers.length,
    };
  }, [architecture, layers.length]);

  // Synchronize scroll across gutter, pre layer, and interactive textarea
  const handleTextareaScroll = (e) => {
    const { scrollTop, scrollLeft } = e.target;
    if (gutterRef.current) gutterRef.current.scrollTop = scrollTop;
    if (preRef.current) {
      preRef.current.scrollTop = scrollTop;
      preRef.current.scrollLeft = scrollLeft;
    }
  };

  const handlePreScroll = (e) => {
    const { scrollTop, scrollLeft } = e.target;
    if (gutterRef.current) gutterRef.current.scrollTop = scrollTop;
    if (textareaRef.current) {
      textareaRef.current.scrollTop = scrollTop;
      textareaRef.current.scrollLeft = scrollLeft;
    }
  };

  // Copy code to clipboard with user feedback
  const handleCopy = async () => {
    try {
      await navigator.clipboard?.writeText(displayCode);
      setCopied(true);
      showToast("PyTorch blueprint copied to clipboard!");
      setTimeout(() => setCopied(false), 2000);
    } catch {
      showToast("Failed to copy code.");
    }
  };

  // Download Python file
  const handleDownload = () => {
    try {
      const filename = `neural_${dataset.name.toLowerCase().replace(/[^a-z0-9]/g, "_")}_model.py`;
      const blob = new Blob([displayCode], { type: "text/x-python;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      showToast(`Downloaded ${filename}!`);
    } catch {
      showToast("Download failed.");
    }
  };

  // Tab key indentation support in textarea
  const handleKeyDown = (e) => {
    if (e.key === "Tab") {
      e.preventDefault();
      const textarea = textareaRef.current;
      if (!textarea) return;
      const start = textarea.selectionStart;
      const end = textarea.selectionEnd;
      const val = textarea.value;
      const updated = val.substring(0, start) + "    " + val.substring(end);
      onCodeChange(updated);
      requestAnimationFrame(() => {
        textarea.selectionStart = textarea.selectionEnd = start + 4;
      });
    }
  };

  // Sync scroll positions and focus textarea when toggling edit mode
  useEffect(() => {
    if (isEditing) {
      if (textareaRef.current && preRef.current) {
        textareaRef.current.scrollTop = preRef.current.scrollTop;
        textareaRef.current.scrollLeft = preRef.current.scrollLeft;
        textareaRef.current.focus();
      }
    } else {
      if (preRef.current && textareaRef.current) {
        preRef.current.scrollTop = textareaRef.current.scrollTop;
        preRef.current.scrollLeft = textareaRef.current.scrollLeft;
      }
    }
  }, [isEditing]);

  return (
    <div className="code-studio">
      {/* Executive Top Banner matching Paper Diagram */}
      <div className="paper-diagram-header code-studio-top-header">
        <div className="paper-title-group">
          <span className="paper-eyebrow">PYTORCH COMPILER · NEURAL BLUEPRINT STUDIO</span>
          <h2>PyTorch Neural Architecture Code (model.py)</h2>
        </div>

        {/* Aggregate Stats Badges matching Paper Diagram */}
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
          <div className="stat-pill">
            <span className="stat-label">WEIGHTS</span>
            <strong className="stat-value" style={{ color: usePretrainedWeights ? "#9de5c1" : "#79b7f4" }}>
              {usePretrainedWeights && currentPreset?.pretrainedWeights ? "Pre-trained" : "Scratch"}
            </strong>
          </div>
        </div>
      </div>

      {/* Secondary Toolbar & Action Controls */}
      <div className="code-studio-toolbar">
        <div className="code-file-tabs">
          <div className="code-file-tab active">
            <svg className="tab-file-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>
            </svg>
            <span className="file-name">model.py</span>
            {codeEdited ? (
              <span className="status-dot modified" title="Custom draft - unsynced" />
            ) : (
              <span className="status-dot synced" title="In sync with architecture" />
            )}
          </div>
          <div className="code-meta-pills">
            <span className="meta-pill">{dataset.name} ({dataset.channels}×{dataset.size}×{dataset.size})</span>
            <span className="meta-pill">{layers.length} Layers</span>
            <span className="meta-pill">{training.optimizer} (lr={training.learningRate})</span>
            <span className="meta-pill status-pill">
              {codeEdited ? (
                <span className="draft-tag">Custom Draft</span>
              ) : (
                <span className="synced-tag">Blueprint Synced</span>
              )}
            </span>
          </div>
        </div>

        <div className="code-studio-actions">
          {/* Live Sync Toggle */}
          <button
            type="button"
            className={`studio-btn sync-toggle-btn ${autoSync ? "sync-on" : "sync-off"}`}
            onClick={onToggleAutoSync}
            title={autoSync ? "Auto-sync is ON: changes sync bidirectionally in real-time" : "Auto-sync is PAUSED: click to re-enable live sync"}
          >
            <span className={`sync-dot ${autoSync ? "live" : "paused"}`} />
            <span>Live Sync: {autoSync ? "ON" : "PAUSED"}</span>
          </button>

          {/* Sync Code to Canvas Graph */}
          <button
            type="button"
            className="studio-btn sync-canvas-btn"
            onClick={() => onSyncToCanvas(displayCode)}
            title="Parse current Python code and update visual architecture layers and settings"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/>
            </svg>
            <span>Sync to Graph</span>
          </button>

          {/* Sync from Graph / Reset Code Button */}
          <button
            type="button"
            className={`studio-btn reset-btn ${codeEdited ? "pulse-action" : ""}`}
            onClick={onResetCode}
            title="Refresh code with current visual architecture blueprint"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/>
              <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
            </svg>
            <span>Sync from Graph</span>
          </button>

          {/* Toggle View / Edit Mode */}
          <button
            type="button"
            className={`studio-btn mode-btn ${isEditing ? "is-editing" : ""}`}
            onClick={() => setIsEditing(!isEditing)}
            title={isEditing ? "Lock edit mode and switch to view" : "Edit Python code directly with live syntax highlighting"}
          >
            {isEditing ? (
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>
              </svg>
            ) : (
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>
              </svg>
            )}
            <span>{isEditing ? "Locked View" : "Edit Code"}</span>
          </button>

          {/* Copy Button */}
          <button
            type="button"
            className="studio-btn copy-btn"
            onClick={handleCopy}
            title="Copy code to clipboard"
          >
            {copied ? (
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#9de5c1" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="20 6 9 17 4 12"/>
              </svg>
            ) : (
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
              </svg>
            )}
            <span>{copied ? "Copied" : "Copy"}</span>
          </button>

          {/* Export File Button */}
          <button
            type="button"
            className="studio-btn download-btn"
            onClick={handleDownload}
            title="Download .py file"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
            <span>Export .py</span>
          </button>

          {/* Back to Canvas */}
          <button
            type="button"
            className="studio-btn back-btn"
            onClick={onBackToArchitecture}
            title="Return to visual canvas"
          >
            <span>Canvas Flow</span>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>
            </svg>
          </button>
        </div>
      </div>

      {/* Unsynced Notification Banner if user edited code and live sync is paused */}
      {codeEdited && !autoSync && (
        <div className="code-unsynced-banner">
          <div className="banner-message">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#f3c979" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
            <div>
              <strong>Manual Code Edits Active</strong>
              <span>Live sync is paused. You can apply custom edits to the visual canvas or refresh from the architecture graph.</span>
            </div>
          </div>
          <div className="banner-actions">
            <button type="button" className="banner-apply-cta" onClick={() => onSyncToCanvas(displayCode)}>
              Sync to Graph
            </button>
            <button type="button" className="banner-reset-cta" onClick={onResetCode}>
              Sync from Graph
            </button>
          </div>
        </div>
      )}

      {/* Code Display Area: Syntax-Highlighted with Synchronized Live Transparent Editing */}
      <div className="code-display-container">
        {/* Line Numbers Gutter */}
        <div className="code-gutter" ref={gutterRef} aria-hidden="true">
          {lines.map((_, i) => (
            <span key={i} className="line-num">{i + 1}</span>
          ))}
        </div>

        {/* Overlaid High-Performance Code Editor preserving syntax highlight during edit */}
        <div className={`code-editor-wrapper ${isEditing ? "is-editing" : "is-viewing"}`}>
          {/* Syntax-highlighted pre layer */}
          <pre
            className="code-highlighted-pre"
            ref={preRef}
            onScroll={!isEditing ? handlePreScroll : undefined}
            tabIndex={isEditing ? -1 : 0}
            onDoubleClick={() => !isEditing && setIsEditing(true)}
            title={!isEditing ? "Double-click or click 'Edit Code' to edit" : undefined}
          >
            <code
              className="language-python"
              dangerouslySetInnerHTML={{ __html: highlightedHtml }}
            />
          </pre>

          {/* Interactive foreground textarea layer */}
          <textarea
            ref={textareaRef}
            className={`code-textarea-input ${isEditing ? "editing-mode" : "readonly-mode"}`}
            value={displayCode}
            readOnly={!isEditing}
            onChange={(e) => onCodeChange(e.target.value)}
            onKeyDown={handleKeyDown}
            onScroll={isEditing ? handleTextareaScroll : undefined}
            spellCheck="false"
            aria-label="Editable Python Code"
            tabIndex={isEditing ? 0 : -1}
          />
        </div>
      </div>

      {/* Studio Footer Status */}
      <div className="code-studio-footer">
        <div className="footer-info">
          <span>Python 3.10+ · PyTorch 2.x</span>
          <span>{lines.length} lines</span>
          <span>UTF-8</span>
          {isEditing && <span className="editing-indicator">Live Syntax-Highlighted Editing</span>}
        </div>
        <div className="footer-status">
          {autoSync ? (
            <span className="status-clean">Live Sync Active · Visual architecture and code are synchronized</span>
          ) : codeEdited ? (
            <span className="status-warning">Live sync paused · Click &ldquo;Sync to Graph&rdquo; to apply custom code</span>
          ) : (
            <span className="status-clean">Auto-generated from computation graph</span>
          )}
        </div>
      </div>
    </div>
  );
}
