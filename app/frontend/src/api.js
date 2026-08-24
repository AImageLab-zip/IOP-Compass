// Thin wrapper over the viewer API. Every call surfaces the backend's own error
// message, which names the file or field at fault, rather than a generic failure.

const BASE = "/api";

async function request(path, options = {}) {
  const response = await fetch(`${BASE}${path}`, options);
  const text = await response.text();
  let payload;
  try {
    payload = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(`${response.status}: ${text.slice(0, 200)}`);
  }
  if (!response.ok) {
    throw new Error(payload.error ?? `request failed (${response.status})`);
  }
  return payload;
}

export const getConfig = () => request("/config");

export const getHealth = () => request("/health");

export function createCase(files, patientId) {
  const form = new FormData();
  if (patientId) form.append("patient_id", patientId);
  files.forEach((file) => form.append("images", file));
  return request("/cases", { method: "POST", body: form });
}

export const classifyCase = (caseId) =>
  request(`/cases/${caseId}/classify`, { method: "POST" });

export const segmentImage = (caseId, body) =>
  request(`/cases/${caseId}/segment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

export const saveCase = (caseId, images, patientId) =>
  request(`/cases/${caseId}/save`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ images, patient_id: patientId }),
  });
