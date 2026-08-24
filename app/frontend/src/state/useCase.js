// Case state: the uploaded images, their predicted views, their instances, and an
// undo stack over every correction. The old interface persisted only the final
// state, so a mis-click cost the clinician the whole crown; here every edit is a
// new immutable snapshot and Ctrl+Z walks back through them.

import { useCallback, useRef, useState } from "react";
import * as api from "../api";

const HISTORY_LIMIT = 60;

const emptyCase = {
  caseId: null,
  patientId: "",
  images: [],
  constrained: false,
};

/** Instances per image id, the only thing undo/redo moves through. */
const emptyInstances = {};

export function useCase() {
  const [caseState, setCaseState] = useState(emptyCase);
  const [instances, setInstances] = useState(emptyInstances);
  const [activeImageId, setActiveImageId] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState(null);
  const [timings, setTimings] = useState({});
  const [segmenter, setSegmenter] = useState("sat");
  const [postprocess, setPostprocess] = useState(true);

  const past = useRef([]);
  const future = useRef([]);

  const commit = useCallback((next) => {
    setInstances((current) => {
      past.current = [...past.current, current].slice(-HISTORY_LIMIT);
      future.current = [];
      return typeof next === "function" ? next(current) : next;
    });
  }, []);

  const undo = useCallback(() => {
    setInstances((current) => {
      const previous = past.current.at(-1);
      if (previous === undefined) return current;
      past.current = past.current.slice(0, -1);
      future.current = [current, ...future.current];
      return previous;
    });
  }, []);

  const redo = useCallback(() => {
    setInstances((current) => {
      const [next, ...rest] = future.current;
      if (next === undefined) return current;
      future.current = rest;
      past.current = [...past.current, current];
      return next;
    });
  }, []);

  /** Replace the active image's instances, recording an undo step. */
  const setActiveInstances = useCallback(
    (updater) => {
      if (!activeImageId) return;
      commit((current) => ({
        ...current,
        [activeImageId]: (typeof updater === "function"
          ? updater(current[activeImageId] ?? [])
          : updater
        ).map((instance, index) => ({
          ...instance,
          instance_id: instance.instance_id ?? `i${index}`,
        })),
      }));
    },
    [activeImageId, commit]
  );

  const run = useCallback(async (label, fn) => {
    setBusy(label);
    setError(null);
    try {
      return await fn();
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setBusy(null);
    }
  }, []);

  /**
   * Upload, then classify, in one action.
   *
   * They are chained because the clinician has no reason to look at an unclassified
   * image: the view decides which SegmentAnyTooth detector runs, so an image without
   * one cannot be segmented at all.
   */
  const uploadAndClassify = useCallback(
    (files, patientId) =>
      run("Uploading and classifying", async () => {
        const created = await api.createCase(files, patientId);
        const classified = await api.classifyCase(created.case_id);
        const byId = Object.fromEntries(
          classified.predictions.map((p) => [p.image_id, p])
        );
        const images = created.images.map((image) => ({
          ...image,
          view: byId[image.image_id]?.view ?? null,
          viewLabel: byId[image.image_id]?.view_label ?? null,
          confidence: byId[image.image_id]?.confidence ?? 0,
          status: "classified",
        }));

        past.current = [];
        future.current = [];
        setInstances(emptyInstances);
        setSelectedId(null);
        setCaseState({
          caseId: created.case_id,
          patientId: created.patient_id,
          images,
          constrained: classified.constrained,
        });
        setActiveImageId(images[0]?.image_id ?? null);
        return images;
      }),
    [run]
  );

  const segmentOne = useCallback(
    (imageId) =>
      run("Segmenting", async () => {
        const image = caseState.images.find((i) => i.image_id === imageId);
        if (!image?.view) throw new Error("image has no view yet");

        setCaseState((state) => ({
          ...state,
          images: state.images.map((i) =>
            i.image_id === imageId ? { ...i, status: "segmenting" } : i
          ),
        }));

        const result = await api.segmentImage(caseState.caseId, {
          image_id: imageId,
          view: image.view,
          segmenter,
          postprocess,
        });

        commit((current) => ({ ...current, [imageId]: result.instances }));
        setTimings((current) => ({ ...current, [imageId]: result.timings }));
        setCaseState((state) => ({
          ...state,
          images: state.images.map((i) =>
            i.image_id === imageId ? { ...i, status: "segmented" } : i
          ),
        }));
        return result;
      }),
    [caseState.caseId, caseState.images, commit, postprocess, run, segmenter]
  );

  /** Segment every image in view order; one request at a time, as the GPU is. */
  const segmentAll = useCallback(async () => {
    for (const image of caseState.images) {
      // eslint-disable-next-line no-await-in-loop
      await segmentOne(image.image_id);
    }
  }, [caseState.images, segmentOne]);

  const setImageView = useCallback((imageId, view) => {
    setCaseState((state) => ({
      ...state,
      images: state.images.map((i) =>
        i.image_id === imageId
          ? { ...i, view, confidence: 0, status: "classified" }
          : i
      ),
    }));
  }, []);

  const save = useCallback(
    (patientId) =>
      run("Saving", async () => {
        const payload = caseState.images
          .filter((image) => (instances[image.image_id] ?? []).length > 0)
          .map((image) => ({
            image_id: image.image_id,
            view: image.view,
            instances: instances[image.image_id],
          }));
        if (!payload.length) throw new Error("nothing to save yet");
        const result = await api.saveCase(caseState.caseId, payload, patientId);
        setCaseState((state) => ({ ...state, patientId: result.patient_id }));
        return result;
      }),
    [caseState.caseId, caseState.images, instances, run]
  );

  const activeImage =
    caseState.images.find((i) => i.image_id === activeImageId) ?? null;

  return {
    caseState,
    activeImage,
    activeImageId,
    setActiveImageId,
    instances,
    activeInstances: instances[activeImageId] ?? [],
    setActiveInstances,
    selectedId,
    setSelectedId,
    busy,
    error,
    setError,
    timings: timings[activeImageId] ?? {},
    segmenter,
    setSegmenter,
    postprocess,
    setPostprocess,
    uploadAndClassify,
    segmentOne,
    segmentAll,
    setImageView,
    save,
    undo,
    redo,
    canUndo: past.current.length > 0,
    canRedo: future.current.length > 0,
  };
}
