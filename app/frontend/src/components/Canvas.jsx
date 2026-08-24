import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Circle, Image as KonvaImage, Layer, Line, Stage, Text } from "react-konva";
import { duplicateFdis, fdiColor } from "../lib/fdi";
import {
  addToInstance,
  centroid,
  decimate,
  eraseFromInstance,
  hitTest,
} from "../lib/geometry";

/** Zoom bounds; below 0.05 the photograph is a thumbnail, above 8 it is pixels. */
const MIN_SCALE = 0.05;
const MAX_SCALE = 8;
const ZOOM_STEP = 1.18;

function useImageElement(src) {
  const [element, setElement] = useState(null);
  useEffect(() => {
    if (!src) {
      setElement(null);
      return undefined;
    }
    const image = new window.Image();
    image.src = src;
    image.onload = () => setElement(image);
    return () => {
      image.onload = null;
    };
  }, [src]);
  return element;
}

/** Fit the photograph inside the pane with a small margin, and centre it. */
function fitView(imageWidth, imageHeight, paneWidth, paneHeight) {
  if (!imageWidth || !paneWidth) return { scale: 1, x: 0, y: 0 };
  const scale = Math.min(paneWidth / imageWidth, paneHeight / imageHeight) * 0.94;
  return {
    scale,
    x: (paneWidth - imageWidth * scale) / 2,
    y: (paneHeight - imageHeight * scale) / 2,
  };
}

export default function Canvas({
  image,
  instances,
  selectedId,
  onSelect,
  onInstancesChange,
  tool,
  opacity,
  showLabels,
  busy,
}) {
  const containerRef = useRef(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [view, setView] = useState({ scale: 1, x: 0, y: 0 });
  const [stroke, setStroke] = useState(null);
  const element = useImageElement(image?.url);

  const duplicates = useMemo(() => duplicateFdis(instances), [instances]);

  useEffect(() => {
    const node = containerRef.current;
    if (!node) return undefined;
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setSize({ width, height });
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const reset = useCallback(() => {
    if (!image || !size.width) return;
    setView(fitView(image.width, image.height, size.width, size.height));
  }, [image, size.width, size.height]);

  // Re-fit whenever the pane or the photograph changes, so switching views never
  // leaves the clinician looking at a corner of the previous image's zoom.
  useEffect(reset, [reset, image?.image_id]);

  const zoomBy = useCallback(
    (factor, originX, originY) => {
      setView((current) => {
        const scale = Math.min(
          MAX_SCALE,
          Math.max(MIN_SCALE, current.scale * factor)
        );
        const cx = originX ?? size.width / 2;
        const cy = originY ?? size.height / 2;
        const ratio = scale / current.scale;
        return {
          scale,
          x: cx - (cx - current.x) * ratio,
          y: cy - (cy - current.y) * ratio,
        };
      });
    },
    [size.width, size.height]
  );

  const toImageCoords = (stage) => {
    const pointer = stage.getPointerPosition();
    if (!pointer) return null;
    return {
      x: (pointer.x - view.x) / view.scale,
      y: (pointer.y - view.y) / view.scale,
    };
  };

  const handleWheel = (event) => {
    event.evt.preventDefault();
    const stage = event.target.getStage();
    const pointer = stage.getPointerPosition();
    zoomBy(event.evt.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP, pointer?.x, pointer?.y);
  };

  const handleDown = (event) => {
    const stage = event.target.getStage();
    const point = toImageCoords(stage);
    if (!point) return;

    if (tool === "select") {
      const hit = [...instances].reverse().find((i) => hitTest(i, point.x, point.y));
      onSelect(hit?.instance_id ?? null);
      return;
    }
    if (tool === "draw" || tool === "add" || tool === "erase") {
      setStroke({ tool, points: [point.x, point.y] });
    }
  };

  const handleMove = (event) => {
    if (!stroke) return;
    const point = toImageCoords(event.target.getStage());
    if (!point) return;
    setStroke((current) => ({
      ...current,
      points: [...current.points, point.x, point.y],
    }));
  };

  const handleUp = () => {
    if (!stroke) return;
    const path = decimate(stroke.points, 2 / view.scale);
    setStroke(null);
    if (path.length < 6) return;

    if (stroke.tool === "draw") {
      // A fresh instance with no FDI: the clinician assigns the number next, and
      // an unnumbered instance is flagged in the list until they do.
      const created = {
        instance_id: `n${Date.now().toString(36)}`,
        fdi: null,
        score: 1,
        contours: [path],
      };
      onInstancesChange([...instances, created]);
      onSelect(created.instance_id);
      return;
    }

    const target = instances.find((i) => i.instance_id === selectedId);
    if (!target) return;
    const next =
      stroke.tool === "add"
        ? addToInstance(target, path)
        : eraseFromInstance(target, path);
    onInstancesChange(
      next
        ? instances.map((i) => (i.instance_id === selectedId ? next : i))
        : instances.filter((i) => i.instance_id !== selectedId)
    );
    if (!next) onSelect(null);
  };

  if (!image) return null;

  const labelSize = Math.max(9, 15 / view.scale);

  return (
    <div className="canvas" ref={containerRef}>
      <Stage
        width={size.width}
        height={size.height}
        scaleX={view.scale}
        scaleY={view.scale}
        x={view.x}
        y={view.y}
        draggable={tool === "pan"}
        onDragEnd={(event) =>
          setView((current) => ({
            ...current,
            x: event.target.x(),
            y: event.target.y(),
          }))
        }
        onWheel={handleWheel}
        onMouseDown={handleDown}
        onMouseMove={handleMove}
        onMouseUp={handleUp}
        onMouseLeave={handleUp}
        style={{ cursor: tool === "pan" ? "grab" : "crosshair" }}
      >
        <Layer listening={false}>
          {element && <KonvaImage image={element} />}
        </Layer>

        <Layer>
          {instances.map((instance) => {
            const color = instance.fdi ? fdiColor(instance.fdi) : "#94a3b8";
            const selected = instance.instance_id === selectedId;
            return (instance.contours ?? []).map((contour, index) => (
              <Line
                key={`${instance.instance_id}-${index}`}
                points={contour}
                closed
                fill={color}
                opacity={selected ? Math.min(1, opacity + 0.22) : opacity}
                stroke={selected ? "#ffffff" : color}
                strokeWidth={(selected ? 2.4 : 1.1) / view.scale}
                listening={false}
              />
            ));
          })}
        </Layer>

        {showLabels && (
          <Layer listening={false}>
            {instances.map((instance) => {
              const { x, y } = centroid(instance);
              const flagged = instance.fdi == null || duplicates.has(instance.fdi);
              return (
                <Text
                  key={`label-${instance.instance_id}`}
                  x={x - labelSize}
                  y={y - labelSize / 2}
                  text={instance.fdi ? String(instance.fdi) : "?"}
                  fontSize={labelSize}
                  fontStyle="bold"
                  fill={flagged ? "#fbbf24" : "#ffffff"}
                  shadowColor="#000000"
                  shadowBlur={4 / view.scale}
                  shadowOpacity={0.95}
                />
              );
            })}
          </Layer>
        )}

        {stroke && (
          <Layer listening={false}>
            <Line
              points={stroke.points}
              closed={stroke.tool === "draw"}
              stroke={stroke.tool === "erase" ? "#f87171" : "#38bdf8"}
              strokeWidth={2 / view.scale}
              dash={[6 / view.scale, 4 / view.scale]}
              fill={stroke.tool === "draw" ? "rgba(56,189,248,0.16)" : undefined}
            />
          </Layer>
        )}
      </Stage>

      {busy && (
        <div className="canvas__busy">
          <span className="spinner" />
          {busy}…
        </div>
      )}

      <div className="canvas__zoom">
        <button type="button" onClick={() => zoomBy(1 / ZOOM_STEP)} title="Zoom out">
          −
        </button>
        <span>{Math.round(view.scale * 100)}%</span>
        <button type="button" onClick={() => zoomBy(ZOOM_STEP)} title="Zoom in">
          +
        </button>
        <button type="button" onClick={reset} title="Fit to window">
          Fit
        </button>
      </div>
    </div>
  );
}
