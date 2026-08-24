import { useRef, useState } from "react";
import { VIEW_LABEL, VIEW_ORDER } from "../lib/fdi";

const STATUS_CLASS = {
  classified: "dot--pending",
  segmenting: "dot--running",
  segmented: "dot--done",
  error: "dot--error",
};

/** Sort by clinical view order, not by upload order, so the strip always reads the same. */
function ordered(images) {
  return [...images].sort(
    (a, b) => VIEW_ORDER.indexOf(a.view) - VIEW_ORDER.indexOf(b.view)
  );
}

function UploadZone({ onFiles, compact }) {
  const inputRef = useRef(null);
  const [over, setOver] = useState(false);

  const handle = (fileList) => {
    const files = [...fileList].filter((f) => f.type.startsWith("image/"));
    if (files.length) onFiles(files);
  };

  return (
    <div
      className={`dropzone ${over ? "dropzone--over" : ""}`}
      onDragOver={(event) => {
        event.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(event) => {
        event.preventDefault();
        setOver(false);
        handle(event.dataTransfer.files);
      }}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") inputRef.current?.click();
      }}
    >
      <strong>{compact ? "Add photographs" : "Drop photographs here"}</strong>
      {!compact && <span>or click to choose · the five standard views</span>}
      <input
        ref={inputRef}
        className="visually-hidden"
        type="file"
        accept="image/*"
        multiple
        onChange={(event) => {
          handle(event.target.files);
          event.target.value = "";
        }}
      />
    </div>
  );
}

export default function ViewStrip({
  images,
  activeImageId,
  onSelect,
  onFiles,
  instances,
  constrained,
}) {
  return (
    <aside className="pane pane--left">
      <div className="pane__title">
        <span>Views</span>
        {constrained && <span title="patient-set constrained assignment">1:1</span>}
      </div>

      <div className="pane__body">
        <div className="views">
          {ordered(images).map((image) => {
            const count = (instances[image.image_id] ?? []).length;
            return (
              <button
                key={image.image_id}
                type="button"
                className={`view ${
                  image.image_id === activeImageId ? "view--active" : ""
                }`}
                onClick={() => onSelect(image.image_id)}
              >
                <img className="view__thumb" src={image.url} alt="" />
                <span className="view__meta">
                  <span className="view__name">
                    {VIEW_LABEL[image.view] ?? "Unclassified"}
                  </span>
                  <span className="view__sub">
                    <span className={`dot ${STATUS_CLASS[image.status] ?? ""}`} />
                    {count > 0
                      ? `${count} teeth`
                      : image.confidence
                        ? `${(image.confidence * 100).toFixed(1)}%`
                        : "—"}
                  </span>
                </span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="pane__footer">
        {images.length > 0 && <CaseSummary images={images} instances={instances} />}
        <UploadZone onFiles={onFiles} compact={images.length > 0} />
      </div>
    </aside>
  );
}

/** Where the case stands overall, so progress is visible without opening each view. */
function CaseSummary({ images, instances }) {
  const segmented = images.filter(
    (image) => (instances[image.image_id] ?? []).length > 0
  ).length;
  const teeth = Object.values(instances).reduce(
    (sum, list) => sum + list.length,
    0
  );
  const unnumbered = Object.values(instances).reduce(
    (sum, list) => sum + list.filter((i) => i.fdi == null).length,
    0
  );

  return (
    <dl className="summary">
      <div>
        <dt>Views segmented</dt>
        <dd>
          {segmented}/{images.length}
        </dd>
      </div>
      <div>
        <dt>Teeth found</dt>
        <dd>{teeth}</dd>
      </div>
      {unnumbered > 0 && (
        <div>
          <dt>Without an FDI</dt>
          <dd className="summary__warn">{unnumbered}</dd>
        </div>
      )}
    </dl>
  );
}

export { UploadZone };
