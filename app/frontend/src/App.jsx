import { useCallback, useEffect, useMemo, useState } from "react";
import * as api from "./api";
import Canvas from "./components/Canvas";
import Header from "./components/Header";
import InstancePanel from "./components/InstancePanel";
import StatusBar from "./components/StatusBar";
import Toolbar from "./components/Toolbar";
import ViewStrip, { UploadZone } from "./components/ViewStrip";
import { duplicateFdis } from "./lib/fdi";
import { centroid, mergeInstances, splitInstance } from "./lib/geometry";
import { useCase } from "./state/useCase";

const KEY_TO_TOOL = { v: "select", h: "pan", n: "draw", a: "add", e: "erase" };

export default function App() {
  const state = useCase();
  const [config, setConfig] = useState(null);
  const [configError, setConfigError] = useState(null);
  const [tool, setTool] = useState("select");
  const [opacity, setOpacity] = useState(0.45);
  const [showLabels, setShowLabels] = useState(true);
  const [notice, setNotice] = useState(null);
  const [patientDraft, setPatientDraft] = useState("");

  useEffect(() => {
    api
      .getConfig()
      .then((payload) => {
        setConfig(payload);
        if (payload.default_segmenter) state.setSegmenter(payload.default_segmenter);
      })
      .catch((err) => setConfigError(`backend unreachable: ${err.message}`));
    // Runs once: the backend's model set does not change while the page is open.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const {
    caseState,
    activeImage,
    activeImageId,
    setActiveImageId,
    instances,
    activeInstances,
    setActiveInstances,
    selectedId,
    setSelectedId,
    busy,
    error,
    setError,
    timings,
    segmenter,
    setSegmenter,
    postprocess,
    setPostprocess,
    uploadAndClassify,
    segmentOne,
    segmentAll,
    save,
    undo,
    redo,
    canUndo,
    canRedo,
  } = state;

  const duplicates = useMemo(
    () => duplicateFdis(activeInstances),
    [activeInstances]
  );
  const unnumbered = activeInstances.filter((i) => i.fdi == null).length;
  const selected =
    activeInstances.find((i) => i.instance_id === selectedId) ?? null;

  const handleFiles = useCallback(
    async (files) => {
      setNotice(null);
      const uploaded = await uploadAndClassify(files, patientDraft);
      if (uploaded) setNotice(`${uploaded.length} image(s) classified`);
    },
    [patientDraft, uploadAndClassify]
  );

  const setFdi = useCallback(
    (instanceId, fdi) => {
      setActiveInstances((current) =>
        current.map((i) => (i.instance_id === instanceId ? { ...i, fdi } : i))
      );
    },
    [setActiveInstances]
  );

  const deleteSelected = useCallback(() => {
    if (!selectedId) return;
    setActiveInstances((current) =>
      current.filter((i) => i.instance_id !== selectedId)
    );
    setSelectedId(null);
  }, [selectedId, setActiveInstances, setSelectedId]);

  const splitSelected = useCallback(() => {
    if (!selected) return;
    const parts = splitInstance(selected);
    if (!parts) return;
    setActiveInstances((current) =>
      current.flatMap((i) => (i.instance_id === selected.instance_id ? parts : i))
    );
    setSelectedId(parts[0].instance_id);
  }, [selected, setActiveInstances, setSelectedId]);

  /** Merge with the nearest other instance — the usual fix for a split crown. */
  const mergeSelected = useCallback(() => {
    if (!selected || activeInstances.length < 2) return;
    const origin = centroid(selected);
    const nearest = activeInstances
      .filter((i) => i.instance_id !== selected.instance_id)
      .map((i) => {
        const other = centroid(i);
        return { instance: i, distance: Math.hypot(other.x - origin.x, other.y - origin.y) };
      })
      .sort((a, b) => a.distance - b.distance)[0];
    if (!nearest) return;

    const merged = mergeInstances(selected, nearest.instance);
    setActiveInstances((current) =>
      current
        .filter((i) => i.instance_id !== nearest.instance.instance_id)
        .map((i) => (i.instance_id === selected.instance_id ? merged : i))
    );
  }, [activeInstances, selected, setActiveInstances]);

  const handleSave = useCallback(async () => {
    const result = await save(patientDraft || caseState.patientId);
    if (result) setNotice(`saved to ${result.patient_id}`);
  }, [caseState.patientId, patientDraft, save]);

  useEffect(() => {
    const onKey = (event) => {
      if (event.target instanceof HTMLInputElement) return;
      const key = event.key.toLowerCase();

      if ((event.ctrlKey || event.metaKey) && key === "z") {
        event.preventDefault();
        (event.shiftKey ? redo : undo)();
        return;
      }
      if (key === "delete" || key === "backspace") {
        event.preventDefault();
        deleteSelected();
        return;
      }
      if (key === "escape") {
        setSelectedId(null);
        return;
      }
      if (KEY_TO_TOOL[key]) setTool(KEY_TO_TOOL[key]);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [deleteSelected, redo, setSelectedId, undo]);

  // Add and Erase need a selection; dropping it must not strand the clinician in a
  // tool that can no longer do anything.
  useEffect(() => {
    if (!selectedId && (tool === "add" || tool === "erase")) setTool("select");
  }, [selectedId, tool]);

  const segmenterOptions = config?.segmenters ?? [];
  const segmenterLabel =
    segmenterOptions.find((s) => s.value === segmenter)?.label ?? segmenter;
  const cell = segmenterOptions.find((s) => s.value === segmenter)?.cell;
  const hasCase = Boolean(caseState.caseId && activeImageId);

  return (
    <div className="app">
      <Header
        patientId={patientDraft || caseState.patientId}
        onPatientIdChange={setPatientDraft}
        activeImage={activeImage}
        segmenterLabel={segmenterLabel}
        postprocess={postprocess}
        hasInstances={activeInstances.length > 0}
      />

      <div className="workspace">
        <ViewStrip
          images={caseState.images}
          activeImageId={activeImageId}
          onSelect={(id) => {
            setActiveImageId(id);
            setSelectedId(null);
          }}
          onFiles={handleFiles}
          instances={instances}
          constrained={caseState.constrained}
        />

        {activeImage ? (
          <div className="stage">
            <Canvas
              image={activeImage}
              instances={activeInstances}
              selectedId={selectedId}
              onSelect={setSelectedId}
              onInstancesChange={setActiveInstances}
              tool={tool}
              opacity={opacity}
              showLabels={showLabels}
              busy={busy}
            />
            <Toolbar
              tool={tool}
              onToolChange={setTool}
              selectedId={selectedId}
              onSplit={splitSelected}
              onMerge={mergeSelected}
              onDelete={deleteSelected}
              canSplit={Boolean(selected && (selected.contours ?? []).length > 1)}
              canMerge={Boolean(selected && activeInstances.length > 1)}
              onUndo={undo}
              onRedo={redo}
              canUndo={canUndo}
              canRedo={canRedo}
            />
          </div>
        ) : (
          <div className="canvas">
            <div className="canvas__empty">
              <h2>Upload a patient&rsquo;s photographs</h2>
              <p>
                Each image is assigned its clinical view automatically, then
                segmented into FDI-numbered tooth instances by SegmentAnyTooth on
                the full image with the benchmark&rsquo;s post-processing. Correct
                whatever the model got wrong and save.
              </p>
              <div style={{ width: 320 }}>
                <UploadZone onFiles={handleFiles} />
              </div>
              {busy && (
                <span style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <span className="spinner" />
                  {busy}…
                </span>
              )}
            </div>
          </div>
        )}

        <InstancePanel
          instances={activeInstances}
          selectedId={selectedId}
          onSelect={setSelectedId}
          onFdiChange={setFdi}
          segmenters={segmenterOptions}
          segmenter={segmenter}
          onSegmenterChange={setSegmenter}
          postprocess={postprocess}
          onPostprocessChange={setPostprocess}
          opacity={opacity}
          onOpacityChange={setOpacity}
          showLabels={showLabels}
          onShowLabelsChange={setShowLabels}
          onSegment={() => segmentOne(activeImageId)}
          onSegmentAll={segmentAll}
          onSave={handleSave}
          busy={busy}
          hasCase={hasCase}
        />
      </div>

      <StatusBar
        instanceCount={activeInstances.length}
        duplicateCount={duplicates.size}
        unnumberedCount={unnumbered}
        timings={timings}
        cell={cell}
        deviceName={config?.device_name}
        imageSize={
          activeImage ? `${activeImage.width}×${activeImage.height}` : null
        }
        error={configError ?? error}
        notice={notice}
      />
    </div>
  );
}
