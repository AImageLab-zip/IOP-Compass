const TOOLS = [
  { id: "select", label: "Select", hint: "V — click an instance" },
  { id: "pan", label: "Pan", hint: "H — drag the image" },
  { id: "draw", label: "New", hint: "N — trace a missed tooth" },
  { id: "add", label: "Add", hint: "A — extend the selected instance" },
  { id: "erase", label: "Erase", hint: "E — trim the selected instance" },
];

export default function Toolbar({
  tool,
  onToolChange,
  selectedId,
  onSplit,
  onMerge,
  onDelete,
  canMerge,
  canSplit,
  onUndo,
  onRedo,
  canUndo,
  canRedo,
}) {
  return (
    <div className="canvas__toolbar">
      {TOOLS.map((item) => (
        <button
          key={item.id}
          type="button"
          title={item.hint}
          className={`tool ${tool === item.id ? "tool--active" : ""}`}
          // Add and Erase act on the selected instance, so they are unreachable
          // until there is one; without this the click silently does nothing.
          disabled={(item.id === "add" || item.id === "erase") && !selectedId}
          onClick={() => onToolChange(item.id)}
        >
          {item.label}
        </button>
      ))}

      <span className="tool__sep" />

      <button
        type="button"
        className="tool"
        onClick={onSplit}
        disabled={!canSplit}
        title="Split a merged instance into its parts"
      >
        Split
      </button>
      <button
        type="button"
        className="tool"
        onClick={onMerge}
        disabled={!canMerge}
        title="Merge the selected instance with the nearest one"
      >
        Merge
      </button>
      <button
        type="button"
        className="tool"
        onClick={onDelete}
        disabled={!selectedId}
        title="Delete the selected instance (Del)"
      >
        Delete
      </button>

      <span className="tool__sep" />

      <button
        type="button"
        className="tool"
        onClick={onUndo}
        disabled={!canUndo}
        title="Undo (Ctrl+Z)"
      >
        Undo
      </button>
      <button
        type="button"
        className="tool"
        onClick={onRedo}
        disabled={!canRedo}
        title="Redo (Ctrl+Shift+Z)"
      >
        Redo
      </button>
    </div>
  );
}
