import { QUADRANTS } from "../lib/fdi";

function Item({ label, value }) {
  return (
    <span className="status__item">
      <strong>{label}</strong>
      {value}
    </span>
  );
}

/** Per-stage timings are shown because the manuscript reports runtime per image. */
export default function StatusBar({
  instanceCount,
  duplicateCount,
  unnumberedCount,
  timings,
  cell,
  deviceName,
  imageSize,
  error,
  notice,
}) {
  if (error) {
    return (
      <footer className="status status--error">
        <Item label="error" value={error} />
      </footer>
    );
  }

  const stages = Object.entries(timings ?? {});
  const total = stages.reduce((sum, [, value]) => sum + value, 0);

  return (
    <footer className="status">
      <Item label="instances" value={instanceCount} />
      {duplicateCount > 0 && (
        <>
          <span className="status__sep">·</span>
          <Item label="duplicate FDI" value={duplicateCount} />
        </>
      )}
      {unnumberedCount > 0 && (
        <>
          <span className="status__sep">·</span>
          <Item label="unnumbered" value={unnumberedCount} />
        </>
      )}
      {stages.map(([name, value]) => (
        <span key={name} className="status__item">
          <span className="status__sep">·</span>
          <strong>{name}</strong>
          {value.toFixed(2)}s
        </span>
      ))}
      {total > 0 && (
        <>
          <span className="status__sep">·</span>
          <Item label="total" value={`${total.toFixed(2)}s`} />
        </>
      )}
      {notice && (
        <>
          <span className="status__sep">·</span>
          <span>{notice}</span>
        </>
      )}

      <span className="status__right">
        {QUADRANTS.map((quadrant) => (
          <span key={quadrant.id} className="legend__item">
            <span
              className="legend__swatch"
              style={{ background: quadrant.color }}
            />
            Q{quadrant.id}
          </span>
        ))}
        {imageSize && <Item label="" value={imageSize} />}
        {cell && <Item label="" value={cell} />}
        {deviceName && <Item label="" value={deviceName} />}
      </span>
    </footer>
  );
}
