// Polygon geometry for the editing tools. Contours are flat [x0,y0,x1,y1,...]
// arrays in the uploaded image's coordinates — the same layout the backend sends
// and expects back, so nothing here has to convert between resolutions.

export function toPairs(flat) {
  const pairs = [];
  for (let i = 0; i + 1 < flat.length; i += 2) pairs.push([flat[i], flat[i + 1]]);
  return pairs;
}

export const toFlat = (pairs) => pairs.flat();

export function polygonArea(flat) {
  const pts = toPairs(flat);
  if (pts.length < 3) return 0;
  let sum = 0;
  for (let i = 0; i < pts.length; i += 1) {
    const [x1, y1] = pts[i];
    const [x2, y2] = pts[(i + 1) % pts.length];
    sum += x1 * y2 - x2 * y1;
  }
  return Math.abs(sum) / 2;
}

export const instanceArea = (instance) =>
  (instance.contours ?? []).reduce((sum, c) => sum + polygonArea(c), 0);

export function pointInPolygon(x, y, flat) {
  const pts = toPairs(flat);
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i, i += 1) {
    const [xi, yi] = pts[i];
    const [xj, yj] = pts[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

export const hitTest = (instance, x, y) =>
  (instance.contours ?? []).some((c) => pointInPolygon(x, y, c));

export function bbox(instance) {
  const pts = (instance.contours ?? []).flatMap(toPairs);
  if (!pts.length) return { minX: 0, minY: 0, maxX: 0, maxY: 0 };
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  return {
    minX: Math.min(...xs),
    minY: Math.min(...ys),
    maxX: Math.max(...xs),
    maxY: Math.max(...ys),
  };
}

export function centroid(instance) {
  let area = 0;
  let cx = 0;
  let cy = 0;
  (instance.contours ?? []).forEach((flat) => {
    const a = polygonArea(flat);
    if (a <= 0) return;
    const pts = toPairs(flat);
    const mx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
    const my = pts.reduce((s, p) => s + p[1], 0) / pts.length;
    cx += mx * a;
    cy += my * a;
    area += a;
  });
  if (area > 0) return { x: cx / area, y: cy / area };
  const { minX, minY, maxX, maxY } = bbox(instance);
  return { x: (minX + maxX) / 2, y: (minY + maxY) / 2 };
}

/**
 * Drop vertices closer together than `tolerance`, so a freehand trace becomes a
 * polygon the browser can drag without dropping frames.
 */
export function decimate(flat, tolerance = 3) {
  const pts = toPairs(flat);
  if (pts.length < 3) return flat;
  const kept = [pts[0]];
  pts.slice(1).forEach(([x, y]) => {
    const [px, py] = kept[kept.length - 1];
    if (Math.hypot(x - px, y - py) >= tolerance) kept.push([x, y]);
  });
  return kept.length >= 3 ? toFlat(kept) : flat;
}

/** Vertices of `flat` that fall inside `region`, as an index set. */
function verticesInside(flat, region) {
  const pts = toPairs(flat);
  const inside = new Set();
  pts.forEach(([x, y], i) => {
    if (pointInPolygon(x, y, region)) inside.add(i);
  });
  return inside;
}

/**
 * Erase `region` from a contour by dropping the vertices it covers.
 *
 * This is a vertex-level approximation, not a boolean polygon operation: it keeps
 * the editing surface dependency-free and responsive, and a clinician correcting a
 * crown boundary works at a scale where the difference is not visible. A contour
 * left with fewer than three vertices is dropped.
 */
export function erasePolygon(flat, region) {
  const pts = toPairs(flat);
  const covered = verticesInside(flat, region);
  if (covered.size === 0) return flat;
  const kept = pts.filter((_, i) => !covered.has(i));
  return kept.length >= 3 ? toFlat(kept) : null;
}

export function eraseFromInstance(instance, region) {
  const contours = (instance.contours ?? [])
    .map((c) => erasePolygon(c, region))
    .filter(Boolean);
  return contours.length ? { ...instance, contours } : null;
}

export function addToInstance(instance, region) {
  return { ...instance, contours: [...(instance.contours ?? []), region] };
}

export function mergeInstances(a, b) {
  return { ...a, contours: [...(a.contours ?? []), ...(b.contours ?? [])] };
}

/**
 * Split a multi-contour instance into one instance per contour.
 *
 * A merge error in these predictions is two crowns in one instance, which
 * post-processing already emits as separate components where it can; this handles
 * what is left. A single-contour instance cannot be split this way.
 */
export function splitInstance(instance) {
  const contours = instance.contours ?? [];
  if (contours.length < 2) return null;
  return contours.map((contour, i) => ({
    ...instance,
    instance_id: `${instance.instance_id}s${i}`,
    contours: [contour],
    fdi: i === 0 ? instance.fdi : null,
  }));
}
