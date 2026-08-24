// FDI two-digit notation: first digit is the quadrant (1 upper right, 2 upper left,
// 3 lower left, 4 lower right), second is the tooth, 1 at the midline to 8 distally.
// Codes 19, 20, 29, 30, 39, 40 do not exist; the vocabulary below is the whole
// permanent dentition and nothing else.

export const QUADRANTS = [
  { id: 1, label: "Upper right", color: "#f472b6" },
  { id: 2, label: "Upper left", color: "#a78bfa" },
  { id: 3, label: "Lower left", color: "#38bdf8" },
  { id: 4, label: "Lower right", color: "#fbbf24" },
];

export const FDI_CODES = QUADRANTS.flatMap(({ id }) =>
  Array.from({ length: 8 }, (_, i) => id * 10 + i + 1)
);

const QUADRANT_COLOR = Object.fromEntries(
  QUADRANTS.map(({ id, color }) => [id, color])
);

export const isValidFdi = (fdi) => FDI_CODES.includes(Number(fdi));

export function fdiColor(fdi) {
  const quadrant = Math.floor(Number(fdi) / 10);
  return QUADRANT_COLOR[quadrant] ?? "#94a3b8";
}

export function fdiArch(fdi) {
  const quadrant = Math.floor(Number(fdi) / 10);
  if (quadrant === 1 || quadrant === 2) return "upper";
  if (quadrant === 3 || quadrant === 4) return "lower";
  return "";
}

export const VIEW_ORDER = [
  "frontal",
  "upper_occlusal",
  "lower_occlusal",
  "left_buccal",
  "right_buccal",
];

export const VIEW_LABEL = {
  frontal: "Frontal",
  upper_occlusal: "Upper occlusal",
  lower_occlusal: "Lower occlusal",
  left_buccal: "Left buccal",
  right_buccal: "Right buccal",
};

/** Duplicate FDI codes within one image; a tooth appears at most once per view. */
export function duplicateFdis(instances) {
  const seen = new Map();
  instances.forEach(({ fdi }) => {
    if (fdi == null) return;
    seen.set(fdi, (seen.get(fdi) ?? 0) + 1);
  });
  return new Set([...seen].filter(([, n]) => n > 1).map(([fdi]) => fdi));
}
