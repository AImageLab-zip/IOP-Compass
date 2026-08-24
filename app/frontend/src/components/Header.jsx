import { VIEW_LABEL } from "../lib/fdi";

/**
 * The pipeline chips are the point of this bar: a reader of the manuscript figure
 * should be able to see view classification -> segmenter -> post-processing without
 * the caption explaining it. Each chip lights up only once its stage has run.
 */
function Chip({ label, value, active }) {
  return (
    <div className={`chip ${active ? "chip--active" : "chip--idle"}`}>
      <span className="chip__key">{label}</span>
      <span className="chip__value">{value}</span>
    </div>
  );
}

export default function Header({
  patientId,
  onPatientIdChange,
  activeImage,
  segmenterLabel,
  postprocess,
  hasInstances,
}) {
  const classified = Boolean(activeImage?.view);
  const confidence = activeImage?.confidence
    ? `${(activeImage.confidence * 100).toFixed(1)}%`
    : null;

  return (
    <header className="header">
      <div className="brand">
        <span className="brand__mark">IOP-Compass</span>
        <span className="brand__sub">tooth instance review</span>
      </div>

      <div className="patient">
        <label className="patient__label" htmlFor="patient-id">
          Patient
        </label>
        <input
          id="patient-id"
          className="patient__input"
          type="text"
          value={patientId}
          placeholder="unassigned"
          onChange={(event) => onPatientIdChange(event.target.value)}
        />
      </div>

      <div className="pipeline">
        <Chip
          label="View"
          value={
            classified
              ? `${VIEW_LABEL[activeImage.view] ?? activeImage.view}${
                  confidence ? ` · ${confidence}` : ""
                }`
              : "not classified"
          }
          active={classified}
        />
        <span className="chip__arrow">→</span>
        <Chip label="Segmenter" value={segmenterLabel} active={hasInstances} />
        <span className="chip__arrow">→</span>
        <Chip
          label="Post"
          value={postprocess ? "on" : "off"}
          active={hasInstances && postprocess}
        />
      </div>
    </header>
  );
}
