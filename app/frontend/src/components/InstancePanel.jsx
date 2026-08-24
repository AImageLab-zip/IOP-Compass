import { useMemo } from "react";
import { FDI_CODES, QUADRANTS, duplicateFdis, fdiColor } from "../lib/fdi";
import { instanceArea } from "../lib/geometry";

/**
 * The FDI keypad. Codes already used in this image are dimmed rather than
 * disabled: reassigning a number is exactly how a clinician fixes a swap, and
 * blocking it would make the duplicate impossible to resolve.
 */
function FdiKeypad({ current, taken, onPick }) {
  return (
    <div className="fdi-grid">
      {FDI_CODES.map((code) => (
        <button
          key={code}
          type="button"
          className={[
            taken.has(code) && code !== current ? "is-taken" : "",
            code === current ? "is-current" : "",
          ]
            .filter(Boolean)
            .join(" ")}
          onClick={() => onPick(code)}
          title={taken.has(code) && code !== current ? `${code} already used` : String(code)}
        >
          {code}
        </button>
      ))}
    </div>
  );
}

export default function InstancePanel({
  instances,
  selectedId,
  onSelect,
  onFdiChange,
  segmenters,
  segmenter,
  onSegmenterChange,
  postprocess,
  onPostprocessChange,
  opacity,
  onOpacityChange,
  showLabels,
  onShowLabelsChange,
  onSegment,
  onSegmentAll,
  onSave,
  busy,
  hasCase,
}) {
  const duplicates = useMemo(() => duplicateFdis(instances), [instances]);
  const taken = useMemo(
    () => new Set(instances.map((i) => i.fdi).filter((f) => f != null)),
    [instances]
  );
  const selected = instances.find((i) => i.instance_id === selectedId) ?? null;
  const unnumbered = instances.filter((i) => i.fdi == null).length;

  return (
    <aside className="pane pane--right">
      <div className="pane__title">
        <span>Instances</span>
        <span>{instances.length}</span>
      </div>

      <div className="pane__body">
        {instances.length === 0 ? (
          <div className="field">
            <span className="hint">
              No instances yet. Run the segmenter on this view, or trace a tooth
              with the New tool.
            </span>
          </div>
        ) : (
          <div className="instances">
            {instances.map((instance) => {
              const duplicate =
                instance.fdi != null && duplicates.has(instance.fdi);
              return (
                <button
                  key={instance.instance_id}
                  type="button"
                  className={`instance ${
                    instance.instance_id === selectedId ? "instance--active" : ""
                  }`}
                  onClick={() => onSelect(instance.instance_id)}
                >
                  <span
                    className="instance__swatch"
                    style={{
                      background: instance.fdi ? fdiColor(instance.fdi) : "#64748b",
                    }}
                  />
                  <span
                    className={`instance__fdi ${
                      instance.fdi == null ? "instance__fdi--unset" : ""
                    }`}
                  >
                    {instance.fdi ?? "—"}
                  </span>
                  <span className="instance__meta">
                    {Math.round(instanceArea(instance)).toLocaleString()} px
                  </span>
                  <span className="instance__flag">
                    {duplicate ? "duplicate" : instance.fdi == null ? "no FDI" : ""}
                  </span>
                </button>
              );
            })}
          </div>
        )}
      </div>

      {selected && (
        <div style={{ flex: "none", borderTop: "1px solid var(--border)" }}>
          <div className="field">
            <span className="field__label">FDI of the selected instance</span>
            <span className="hint">
              {selected.fdi
                ? `${selected.fdi} · ${
                    QUADRANTS.find((q) => q.id === Math.floor(selected.fdi / 10))
                      ?.label ?? ""
                  }`
                : "unassigned — pick a code below"}
            </span>
          </div>
          <FdiKeypad
            current={selected.fdi}
            taken={taken}
            onPick={(code) => onFdiChange(selected.instance_id, code)}
          />
        </div>
      )}

      <div style={{ flex: "none", borderTop: "1px solid var(--border)" }}>
        <div className="field">
          <span className="field__label">Segmenter</span>
          <select
            value={segmenter}
            onChange={(event) => onSegmenterChange(event.target.value)}
          >
            {segmenters.map((item) => (
              <option key={item.value} value={item.value}>
                {item.label}
              </option>
            ))}
          </select>
          <span className="hint">
            Runs on the full image; no ROI stage.
          </span>
        </div>

        <div className="field">
          <div className="field__row">
            <label className="field__label" htmlFor="postprocess">
              Post-processing
            </label>
            <input
              id="postprocess"
              type="checkbox"
              checked={postprocess}
              onChange={(event) => onPostprocessChange(event.target.checked)}
              style={{ marginLeft: "auto", width: "auto" }}
            />
          </div>
        </div>

        <div className="field">
          <div className="field__row">
            <span className="field__label">Overlay</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={opacity}
              onChange={(event) => onOpacityChange(Number(event.target.value))}
            />
            <span className="field__value">{Math.round(opacity * 100)}%</span>
          </div>
          <div className="field__row">
            <label className="field__label" htmlFor="labels">
              FDI labels
            </label>
            <input
              id="labels"
              type="checkbox"
              checked={showLabels}
              onChange={(event) => onShowLabelsChange(event.target.checked)}
              style={{ marginLeft: "auto", width: "auto" }}
            />
          </div>
        </div>
      </div>

      <div className="pane__footer" style={{ display: "grid", gap: 6 }}>
        <button
          type="button"
          className="btn btn--primary"
          onClick={onSegment}
          disabled={!hasCase || Boolean(busy)}
        >
          Segment this view
        </button>
        <div style={{ display: "flex", gap: 6 }}>
          <button
            type="button"
            className="btn"
            onClick={onSegmentAll}
            disabled={!hasCase || Boolean(busy)}
          >
            All views
          </button>
          <button
            type="button"
            className="btn"
            onClick={onSave}
            disabled={!hasCase || Boolean(busy) || instances.length === 0}
            title={
              unnumbered
                ? `${unnumbered} instance(s) still have no FDI code`
                : "Write the corrected annotations"
            }
          >
            Save
          </button>
        </div>
      </div>
    </aside>
  );
}
